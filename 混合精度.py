import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, metrics, regularizers, callbacks
import matplotlib.pyplot as plt
import time
from tensorflow.keras.utils import plot_model

# ==============================
# 0. 混合精度設定 (Mixed Precision)
# ==============================
# 1. 匯入混合精度模組
from tensorflow.keras import mixed_precision

# 2. 設定全局政策為 'mixed_float16'，表示除輸出層外大部分計算使用半精度 (float16)
#    這裡會自動在適當層自動套用 float16，而梯度累加等關鍵操作仍保留 float32
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
# 此時再印政策可驗證：應顯示 "mixed_float16"
print("Mixed precision policy:", mixed_precision.global_policy())

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
#    - 注意：因全局政策已設定為混合精度，除了最終輸出層需指定 float32，
#      其餘層會自動使用 float16/float32 混合計算
# ==============================
def build_advanced_cnn():
    weight_decay = 1e-3
    inputs = layers.Input(shape=(32, 32, 3))

    # 第一段卷積區塊
    x = layers.Conv2D(
        64, 3, padding='same',
        kernel_regularizer=regularizers.l2(weight_decay)
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        64, 3, padding='same',
        kernel_regularizer=regularizers.l2(weight_decay)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Dropout(0.3)(x)

    # 第二段卷積區塊
    x = layers.Conv2D(
        128, 3, padding='same',
        kernel_regularizer=regularizers.l2(weight_decay)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        128, 3, padding='same',
        kernel_regularizer=regularizers.l2(weight_decay)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Dropout(0.4)(x)

    # 第三段卷積區塊
    x = layers.Conv2D(
        256, 3, padding='same',
        kernel_regularizer=regularizers.l2(weight_decay)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Dropout(0.5)(x)

    # 全連接區塊
    x = layers.Flatten()(x)
    x = layers.Dense(
        512, activation='relu',
        kernel_regularizer=regularizers.l2(weight_decay)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    # 最後一層輸出層：必須顯式指定 dtype='float32'
    # 否則因為前面使用 float16，logits 會變成 float16，計算 loss 可能數值不穩
    outputs = layers.Dense(100, dtype='float32')(x)

    return models.Model(inputs, outputs, name='advanced_cnn_mixed')

model = build_advanced_cnn()
model.summary()  # 可以看到大部分層的 dtype 是 float16，但最後一層為 float32

# ==============================
# 4. 編譯模型
#    - 混合精度政策下，建議將優化器包裝為 LossScaleOptimizer，
#      以自動放縮 loss 來減少 float16 下溢/溢出的風險
# ==============================
# 1. 建立基礎優化器
base_optimizer = optimizers.Adam(learning_rate=1e-3)

# 2. 使用 LossScaleOptimizer 包裝
optimizer = mixed_precision.LossScaleOptimizer(base_optimizer)

model.compile(
    optimizer=optimizer,
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# ==============================
# 5. 設定 Callback：早停與學習率衰減
# ==============================
early_stop = callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True,
    verbose=1
)
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# ==============================
# 6. 訓練 (with tqdm-like progress bar)
# ==============================
start_time = time.time()
history = model.fit(
    train_ds,
    epochs=100,
    validation_data=val_ds,
    callbacks=[early_stop, reduce_lr],
    verbose=1       # 會顯示進度列和每 epoch 的 metrics
)
train_time = time.time() - start_time

# ==============================
# 7. 測試集評估
# ==============================
test_loss, test_acc = model.evaluate(test_ds, verbose=2)

print(f"\nTraining time: {train_time:.2f} seconds")
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# ==============================
# 8. 繪製 Loss & Accuracy 曲線
# ==============================
epochs_range = range(1, len(history.history['loss']) + 1)

plt.figure(figsize=(14,5))

plt.subplot(1,2,1)
plt.plot(epochs_range, history.history['loss'],     label='Train Loss')
plt.plot(epochs_range, history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs_range, history.history['accuracy'],     label='Train Acc')
plt.plot(epochs_range, history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
