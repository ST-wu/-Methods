import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, metrics, regularizers, callbacks
import matplotlib.pyplot as plt
import numpy as np
import time

# 1. 資料載入與預處理
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
x_train, x_test = x_train.astype('float32') / 255.0, x_test.astype('float32') / 255.0
y_train, y_test = y_train.flatten(), y_test.flatten()

# 切出驗證集
val_split = 0.1
val_size = int(len(x_train) * val_split)
x_val, y_val     = x_train[:val_size], y_train[:val_size]
x_train2, y_train2 = x_train[val_size:], y_train[val_size:]

# 2. 建立 Dataset 管道
batch_size = 128
train_ds = tf.data.Dataset.from_tensor_slices((x_train2, y_train2)) \
                         .shuffle(50000).batch(batch_size).prefetch(2)
val_ds   = tf.data.Dataset.from_tensor_slices((x_val, y_val)) \
                         .batch(batch_size).prefetch(2)
# 測試集我們在後面直接用 numpy array 做迭代
test_ds  = tf.data.Dataset.from_tensor_slices((x_test, y_test)) \
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

    outputs = layers.Dense(100)(x)  # from_logits=True, 後面用 softmax
    return models.Model(inputs, outputs, name='advanced_cnn')

# 建立模型、編譯
model = build_advanced_cnn()
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# 4. Callback：早停與 LR 衰減
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

# 5. 訓練
start_time = time.time()
history = model.fit(
    train_ds,
    epochs=100,
    validation_data=val_ds,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)
train_time = time.time() - start_time

# 6. 一般測試集評估（baseline）
test_loss, test_acc = model.evaluate(test_ds, verbose=2)
print(f"\n[Baseline] Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# 7. 加入「鏡像補救」的自訂評估函式
def evaluate_with_flip(model, x_data, y_data, threshold=0.8, batch_size=128):
    """
    對於每張 x_data[i]：
      1. model.predict 出 logits，softmax 得到 prob1；取 max(prob1) 與 argmax(prob1)
      2. 如果 max(prob1) < threshold，則對該張圖片做水平鏡像再 predict，一樣取 prob2、argmax2
      3. 最終決策：比較 prob1[argmax1] 與 prob2[argmax2]，取機率較大者的標籤做為最終 pred
      4. 若 max(prob1) >= threshold，則只用 prob1 的 argmax1 做最終 pred
    回傳：最終預測的正確率 (accuracy)
    """
    total = len(x_data)
    correct = 0

    # 分 batch 處理以加速
    num_batches = int(np.ceil(total / batch_size))
    for b in range(num_batches):
        start = b * batch_size
        end = min((b+1) * batch_size, total)
        x_batch = x_data[start:end]
        y_batch = y_data[start:end]

        # 第一次推論
        logits1 = model.predict(x_batch, verbose=0)
        probs1 = tf.nn.softmax(logits1, axis=-1).numpy()
        maxp1 = np.max(probs1, axis=-1)          # shape=(batch_size,)
        arg1  = np.argmax(probs1, axis=-1)       # shape=(batch_size,)

        # 建立一個容器存放最終預測
        final_preds = np.copy(arg1)

        # 找出哪些樣本需要第二次鏡像預測 (conf < threshold)
        need_flip_idx = np.where(maxp1 < threshold)[0]
        if len(need_flip_idx) > 0:
            # 對這些需要補救的圖片做水平鏡像
            x_flip = x_batch[need_flip_idx][:, :, ::-1, :]  # 水平翻轉

            logits2 = model.predict(x_flip, verbose=0)
            probs2 = tf.nn.softmax(logits2, axis=-1).numpy()
            maxp2 = np.max(probs2, axis=-1)
            arg2  = np.argmax(probs2, axis=-1)

            # 對於每個 need_flip_idx 的樣本，比較兩次機率，取大者的 argmax 作為最終
            for i_local, global_idx in enumerate(need_flip_idx):
                if maxp2[i_local] > maxp1[global_idx]:
                    final_preds[global_idx] = arg2[i_local]
                # 否則仍保留第一次的 arg1[global_idx]

        # 計算正確數
        correct += np.sum(final_preds == y_batch)

    accuracy = correct / total
    return accuracy

# 以不同 threshold 試算
thresholds = [0.6, 0.7, 0.8, 0.9]
for thr in thresholds:
    acc_flip = evaluate_with_flip(model, x_test, y_test, threshold=thr, batch_size=batch_size)
    print(f"[Flip‐check] threshold={thr:.2f} --> Test Accuracy: {acc_flip:.4f}")

# 8. 繪製 Loss & Accuracy 曲線（同原本程式）
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

# 9. 顯示訓練時間與 baseline 結果
print(f"\nTraining time: {train_time:.2f} seconds")
print(f"[Baseline] Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

"""
[Baseline] Test Loss: 1.6063, Test Accuracy: 0.6603
[Flip‐check] threshold=0.60 --> Test Accuracy: 0.6693
[Flip‐check] threshold=0.70 --> Test Accuracy: 0.6700
[Flip‐check] threshold=0.80 --> Test Accuracy: 0.6705
[Flip‐check] threshold=0.90 --> Test Accuracy: 0.6706
"""
