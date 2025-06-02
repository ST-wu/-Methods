import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, metrics, regularizers, callbacks
import matplotlib.pyplot as plt
import time

# 若要使用 KerasTuner 作為 AutoML / 隨機搜索工具，需先安裝 keras-tuner：
# pip install keras-tuner

from keras_tuner import RandomSearch
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

# 2. 建立 Dataset 管道
batch_size = 128
train_ds = tf.data.Dataset.from_tensor_slices((x_train2, y_train2)) \
                         .shuffle(50000).batch(batch_size).prefetch(2)
val_ds   = tf.data.Dataset.from_tensor_slices((x_val, y_val)) \
                         .batch(batch_size).prefetch(2)
test_ds  = tf.data.Dataset.from_tensor_slices((x_test, y_test)) \
                         .batch(batch_size).prefetch(2)

# ==============================
# 3. 將模型定義成能夠接受「超參數」的函式，以供 KerasTuner 搜索
#    HyperParameters (hp) 物件可定義需要搜索的空間
# ==============================
def build_model_hp(hp):
    """
    使用 KerasTuner 的 HyperParameters 以隨機搜索不同結構和超參數組合。
    這裡範例搜尋：
      - Conv2D 濾波器數量
      - Dense 層單元數
      - Dropout 比例
      - 學習率
    """
    weight_decay = 1e-3
    inputs = layers.Input(shape=(32, 32, 3))

    # 第一層卷積：濾波器數量可選 32、64、128
    filters1 = hp.Choice('filters1', values=[32, 64, 128], default=64)
    x = layers.Conv2D(filters1, 3, padding='same',
                      kernel_regularizer=regularizers.l2(weight_decay))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # 第二層卷積：濾波器數量可選擇範圍與第一層不同
    filters2 = hp.Choice('filters2', values=[64, 128, 256], default=128)
    x = layers.Conv2D(filters2, 3, padding='same',
                      kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool2D()(x)

    # Dropout 比例：在 0～0.5 之間搜尋
    dropout1 = hp.Float('dropout1', min_value=0.2, max_value=0.5, step=0.1, default=0.3)
    x = layers.Dropout(dropout1)(x)

    # 第三層卷積
    filters3 = hp.Choice('filters3', values=[128, 256, 512], default=256)
    x = layers.Conv2D(filters3, 3, padding='same',
                      kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool2D()(x)

    dropout2 = hp.Float('dropout2', min_value=0.3, max_value=0.6, step=0.1, default=0.5)
    x = layers.Dropout(dropout2)(x)

    # 全連接層
    x = layers.Flatten()(x)
    dense_units = hp.Int('dense_units', min_value=256, max_value=1024, step=256, default=512)
    x = layers.Dense(dense_units, activation='relu',
                     kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    dropout3 = hp.Float('dropout3', min_value=0.3, max_value=0.6, step=0.1, default=0.5)
    x = layers.Dropout(dropout3)(x)

    # 輸出層
    outputs = layers.Dense(100)(x)

    model = models.Model(inputs, outputs)
    
    # 學習率可搜尋 1e-4～1e-2 之間的對數空間
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log', default=1e-3)
    optimizer = optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model

# ==============================
# 4. 設定 KerasTuner 的 RandomSearch 或 Hyperband 作為 AutoML/隨機搜索
# ==============================
tuner = RandomSearch(
    build_model_hp,                 # 傳入剛剛定義、使用 hp 的 model builder 函式
    objective='val_accuracy',       # 以驗證準確度作為優化目標
    max_trials=10,                  # 最多嘗試 10 種不同超參數組合
    executions_per_trial=1,         # 每組超參數只訓練一次
    directory='ktuner_dir',         # 儲存搜尋過程結果的資料夾
    project_name='cifar100_tuning'  # 該專案名稱
)

# 顯示搜尋空間摘要
tuner.search_space_summary()

# ==============================
# 5. 開始搜尋超參數：相當於在原本的訓練迴圈上做隨機搜索
#    - 會自動依不同超參數組合建立模型、訓練再評估，並儲存最佳組合
# ==============================
tuner.search(
    train_ds,
    epochs=30,               # 每組超參數最多訓練 30 個 epoch
    validation_data=val_ds,  # 使用驗證集來衡量 val_accuracy
    callbacks=[
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ],
    verbose=1
)

# 列出搜尋結果的摘要
tuner.results_summary()

# ==============================
# 6. 取出最佳模型
# ==============================
best_model = tuner.get_best_models(num_models=1)[0]
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print("\n=== 最佳超參數組合 ===")
print(f"filters1:     {best_hps.get('filters1')}")
print(f"filters2:     {best_hps.get('filters2')}")
print(f"filters3:     {best_hps.get('filters3')}")
print(f"dropout1:     {best_hps.get('dropout1'):.2f}")
print(f"dropout2:     {best_hps.get('dropout2'):.2f}")
print(f"dropout3:     {best_hps.get('dropout3'):.2f}")
print(f"dense_units:  {best_hps.get('dense_units')}")
print(f"learning_rate:{best_hps.get('learning_rate'):.5f}")

# ==============================
# 7. 使用最佳模型繼續訓練或直接評估
# ==============================
start_time = time.time()
history = best_model.fit(
    train_ds,
    epochs=50,               # 可視情況再多訓練一些 epoch
    validation_data=val_ds,
    callbacks=[
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ],
    verbose=1
)
train_time = time.time() - start_time

# 8. 在測試集上評估
test_loss, test_acc = best_model.evaluate(test_ds, verbose=2)
print(f"\nTraining time (fine-tune): {train_time:.2f} seconds")
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# ==============================
# 9. 繪製 Loss & Accuracy 曲線（可選）
# ==============================
epochs_range = range(1, len(history.history['loss']) + 1)

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, history.history['loss'],     label='Train Loss')
plt.plot(epochs_range, history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, history.history['accuracy'],     label='Train Acc')
plt.plot(epochs_range, history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
