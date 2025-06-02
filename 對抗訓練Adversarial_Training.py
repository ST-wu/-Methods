import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, metrics, regularizers, callbacks
import matplotlib.pyplot as plt
import time

# ==============================
# 0. 定義對抗訓練的參數
# ==============================
epsilon = 0.01  # FGSM 擾動強度

# ==============================
# 1. 資料載入與預處理
# ==============================
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
x_train, x_test = x_train.astype('float32') / 255.0, x_test.astype('float32') / 255.0
y_train, y_test = y_train.flatten(), y_test.flatten()

# 切出驗證集
val_split = 0.1
val_size = int(len(x_train) * val_split)
x_val, y_val       = x_train[:val_size], y_train[:val_size]
x_train2, y_train2 = x_train[val_size:], y_train[val_size:]

# ==============================
# 2. 建立 Dataset 管道
# ==============================
batch_size = 128
train_ds = tf.data.Dataset.from_tensor_slices((x_train2, y_train2)) \
                         .shuffle(50000).batch(batch_size).prefetch(2)
val_ds   = tf.data.Dataset.from_tensor_slices((x_val, y_val)) \
                         .batch(batch_size).prefetch(2)
test_ds  = tf.data.Dataset.from_tensor_slices((x_test, y_test)) \
                         .batch(batch_size).prefetch(2)

# ==============================
# 3. 定義增強後的 CNN 模型
# ==============================
def build_advanced_cnn():
    weight_decay = 1e-3
    inputs = layers.Input(shape=(32, 32, 3))
    x = layers.Conv2D(64, 3, padding='same',
                      kernel_regularizer=regularizers.l2(weight_decay))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, 3, padding='same',
                      kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, 3, padding='same',
                      kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, 3, padding='same',
                      kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Conv2D(256, 3, padding='same',
                      kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu',
                     kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(100)(x)
    return models.Model(inputs, outputs, name='advanced_cnn')

base_model = build_advanced_cnn()

# ==============================
# 4. 自訂 Model，加上對抗訓練邏輯，並修正 test_step 的返回格式
# ==============================
class AdvModel(models.Model):
    def __init__(self, base_model, epsilon):
        super().__init__()
        self.model = base_model
        self.epsilon = epsilon

    def compile(self, optimizer, loss, metrics):
        super().compile()
        self.optimizer = optimizer
        # loss 已實例化並含 from_logits=True
        self.loss_fn = loss(from_logits=True)
        # 只保留兩個 train-phase 指標
        self.train_loss = metrics.Mean(name='loss')
        self.train_acc  = metrics.SparseCategoricalAccuracy(name='accuracy')
        # 只保留兩個 test-phase 指標
        self.val_loss = metrics.Mean(name='val_loss')
        self.val_acc  = metrics.SparseCategoricalAccuracy(name='val_accuracy')

    def train_step(self, data):
        x, y = data

        # ------------------------------
        # 1) 計算對抗樣本 (FGSM)
        # ------------------------------
        with tf.GradientTape() as tape_img:
            tape_img.watch(x)
            logits = self.model(x, training=True)
            loss_value = self.loss_fn(y, logits)  # 直接呼叫實例化過的 loss_fn
        grad = tape_img.gradient(loss_value, x)
        x_adv = x + self.epsilon * tf.sign(grad)
        x_adv = tf.clip_by_value(x_adv, 0.0, 1.0)

        # ------------------------------
        # 2) 使用對抗樣本做一次訓練
        # ------------------------------
        with tf.GradientTape() as tape:
            logits_adv = self.model(x_adv, training=True)
            loss_adv = self.loss_fn(y, logits_adv)
        grads = tape.gradient(loss_adv, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # 更新 train-phase 指標
        self.train_loss.update_state(loss_adv)
        self.train_acc.update_state(y, logits_adv)

        return {
            'loss': self.train_loss.result(),
            'accuracy': self.train_acc.result()
        }

    def test_step(self, data):
        x, y = data
        logits = self.model(x, training=False)
        loss_value = self.loss_fn(y, logits)

        # 更新 test-phase 指標，並且返回鍵名跟 train_phase 一致
        self.val_loss.update_state(loss_value)
        self.val_acc.update_state(y, logits)
        return {
            'loss': self.val_loss.result(),
            'accuracy': self.val_acc.result()
        }

# ==============================
# 5. 建立對抗訓練模型並編譯
# ==============================
adv_model = AdvModel(base_model, epsilon=epsilon)

optimizer = optimizers.Adam(learning_rate=1e-3)
adv_model.compile(
    optimizer=optimizer,
    loss=losses.SparseCategoricalCrossentropy,
    metrics=metrics
)

adv_model.model.summary()

# ==============================
# 6. 設定 Callback：早停與學習率衰減，監測 val_accuracy 就能正確運作
# ==============================
early_stop = callbacks.EarlyStopping(
    monitor='val_accuracy',  # 此時“val_accuracy”已存在
    patience=10,
    restore_best_weights=True,
    verbose=1
)
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_accuracy',  # 同上
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# ==============================
# 7. 使用 model.fit 進行對抗訓練
# ==============================
start_time = time.time()
history = adv_model.fit(
    train_ds,
    epochs=100,
    validation_data=val_ds,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)
train_time = time.time() - start_time

# ==============================
# 8. 測試集評估（乾淨樣本）
# ==============================
test_metrics = adv_model.evaluate(test_ds, verbose=2)
print(f"\nTraining time: {train_time:.2f} seconds")
print(f"Test Loss: {test_metrics[0]:.4f}, Test Accuracy: {test_metrics[1]:.4f}")

# ==============================
# 9. 繪製 Loss & Accuracy 曲線
# ==============================
epochs_range = range(1, len(history.history['loss']) + 1)

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, history.history['loss'],     label='Train Loss (adv)')
plt.plot(epochs_range, history.history['val_loss'], label='Val Loss (clean)')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, history.history['accuracy'],     label='Train Acc (adv)')
plt.plot(epochs_range, history.history['val_accuracy'], label='Val Acc (clean)')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
