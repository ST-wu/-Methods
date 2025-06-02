import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, metrics, regularizers, callbacks
import matplotlib.pyplot as plt
import time
from tensorflow.keras.utils import plot_model
# 1. 資料載入與預處理
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
x_train, x_test = x_train.astype('float32')/255.0, x_test.astype('float32')/255.0
y_train, y_test = y_train.flatten(), y_test.flatten()

# 切出驗證集
val_split = 0.1
val_size = int(len(x_train) * val_split)
x_val, y_val   = x_train[:val_size], y_train[:val_size]
x_train2, y_train2 = x_train[val_size:], y_train[val_size:]

# 2. 建立 Dataset 管道
batch_size = 128
train_ds = tf.data.Dataset.from_tensor_slices((x_train2, y_train2))\
                         .shuffle(50000).batch(batch_size).prefetch(2)
val_ds   = tf.data.Dataset.from_tensor_slices((x_val,    y_val   ))\
                         .batch(batch_size).prefetch(2)
test_ds  = tf.data.Dataset.from_tensor_slices((x_test,   y_test  ))\
                         .batch(batch_size).prefetch(2)

# 3. 定義增強後的 CNN 模型
def build_advanced_cnn():
    weight_decay = 1e-3
    inputs = layers.Input(shape=(32,32,3))
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

model = build_advanced_cnn()
model.summary()

# 4. 編譯模型
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    # metrics=[metrics.SparseCategoricalAccuracy()]
    metrics=['accuracy']
)

# 5. 設定 Callback：早停與學習率衰減
early_stop = callbacks.EarlyStopping(
    # monitor='val_sparse_categorical_accuracy',
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True,
    verbose=1
)
reduce_lr = callbacks.ReduceLROnPlateau(
    # monitor='val_sparse_categorical_accuracy',
    monitor='val_accuracy',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# 6. 訓練 (with tqdm-like progress bar)
start_time = time.time()
history = model.fit(
    train_ds,
    epochs=100,
    validation_data=val_ds,
    callbacks=[early_stop, reduce_lr],
    verbose=1       # 會顯示進度列和每 epoch 的 metrics
)
train_time = time.time() - start_time

# 7. 測試集評估
test_loss, test_acc = model.evaluate(test_ds, verbose=2)

print(f"\nTraining time: {train_time:.2f} seconds")
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# 8. 繪製 Loss & Accuracy 曲線
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
