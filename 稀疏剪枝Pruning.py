import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, metrics, regularizers, callbacks
import matplotlib.pyplot as plt
import time

# 新增：匯入 TensorFlow Model Optimization Toolkit，用於剪枝 (Pruning)
import tensorflow_model_optimization as tfmot

# 1. 資料載入與預處理
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
x_train, x_test = x_train.astype('float32')/255.0, x_test.astype('float32')/255.0
y_train, y_test = y_train.flatten(), y_test.flatten()

# 切出驗證集
val_split = 0.1
val_size = int(len(x_train) * val_split)
x_val, y_val       = x_train[:val_size], y_train[:val_size]
x_train2, y_train2 = x_train[val_size:], y_train[val_size:]

# 2. 建立 Dataset 管道
batch_size = 128
train_ds = tf.data.Dataset.from_tensor_slices((x_train2, y_train2))\
                         .shuffle(50000).batch(batch_size).prefetch(2)
val_ds   = tf.data.Dataset.from_tensor_slices((x_val,    y_val   ))\
                         .batch(batch_size).prefetch(2)
test_ds  = tf.data.Dataset.from_tensor_slices((x_test,   y_test  ))\
                         .batch(batch_size).prefetch(2)

# 3. 定義增強後的 CNN 模型（未剪枝版本）
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

# 3a. 套用剪枝：使用 tfmot.sparsity.keras.prune_low_magnitude
def build_pruned_cnn():
    # 定義剪枝 schedule：從訓練開始到結束，逐漸將稀疏率從 0 提升到 50%
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,      # 開始時不剪枝
            final_sparsity=0.5,        # 到訓練結束時剪掉 50% 權重
            begin_step=0,              # 從 step=0 開始
            end_step=np.ceil(len(x_train2) / batch_size).astype(np.int32) * 100  # 將 end_step 設為總步數（假設訓練 100 個 epoch）
        )
    }

    # 先建立原始模型
    base_model = build_advanced_cnn()

    # 利用 prune_low_magnitude 將原始模型轉換為可剪枝版本
    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(base_model, **pruning_params)

    return model_for_pruning

# 4. 建立並顯示剪枝後模型
import numpy as np
model = build_pruned_cnn()
model.summary()  # 可以看到部分層被包裹在 PruneLowMagnitude Wrapper 中

# 5. 編譯模型
# 剪枝與 L2 正則同時使用，損失函數一樣，注意優化器不需做特別改動
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# 6. 設定 Callback：早停、學習率衰減，以及剪枝相關 Callback
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

# 剪枝 Callback：在每個訓練 step 更新剪枝 mask
pruning_callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    # 可以選擇將稀疏率與剪枝資訊寫到 TensorBoard
    tfmot.sparsity.keras.PruningSummaries(log_dir='./pruning_logs', profile_batch=0)
]

# 合併所有 callbacks
all_callbacks = [early_stop, reduce_lr] + pruning_callbacks

# 7. 訓練 (with tqdm-like progress bar)
start_time = time.time()
history = model.fit(
    train_ds,
    epochs=100,
    validation_data=val_ds,
    callbacks=all_callbacks,
    verbose=1
)
train_time = time.time() - start_time

# 8. 在測試前，先「剝除」(strip) 剪枝 wrappers，得到最終的精簡模型
model_for_export = tfmot.sparsity.keras.strip_pruning(model)

# 9. 測試集評估
test_loss, test_acc = model_for_export.evaluate(test_ds, verbose=2)
print(f"\nTraining time: {train_time:.2f} seconds")
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# 10. 繪製 Loss & Accuracy 曲線
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
