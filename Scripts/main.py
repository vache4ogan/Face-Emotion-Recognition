import tensorflow as tf
import numpy as np
import os
from PIL import Image
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


train_dir = 'D:/Projects/Face_emotion/dataset/train/'
test_dir = 'D:/Projects/Face_emotion/dataset/test/'
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']




gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Устанавливаем ограничение: занимать память только по мере необходимости
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU настроен на динамический рост памяти")
    except RuntimeError as e:
        print(e)






def train_data_load(path):
    img_list = []
    target = []
    ind =0 
    for i in emotions:
        files = [f for f in os.listdir(path+i) if f.endswith(('.png', '.jpg', '.jpeg'))]

        for filename in files:
            img_path = os.path.join(path+i, filename)
        
        # Открываем изображение и конвертируем в L (черно-белый режим), на всякий случай
            img = Image.open(img_path).convert('L')
        
        # Превращаем картинку в массив numpy
            img_array = np.array(img)
        
        # Добавляем в наш список
            img_list.append(img_array)
            target.append(ind)
        ind+=1
    
    # Побуждаем Python склеить список в один большой массив (тензор)
    X = np.array(img_list, dtype='float32') / 255.0
    X = np.expand_dims(X, axis=-1)
    y = np.array(target)


    return X, y

def test_data_load(path):
    img_list = []
    target = []
    ind =0 
    for i in emotions:
        files = [f for f in os.listdir(path+i) if f.endswith(('.png', '.jpg', '.jpeg'))]

        for filename in files:
            img_path = os.path.join(path+i, filename)
        
        # Открываем изображение и конвертируем в L (черно-белый режим), на всякий случай
            img = Image.open(img_path).convert('L')
        
        # Превращаем картинку в массив numpy
            img_array = np.array(img)
        
        # Добавляем в наш список
            img_list.append(img_array)
            target.append(ind)
        ind+=1
    
    # Побуждаем Python склеить список в один большой массив (тензор)
    X = np.array(img_list, dtype='float32') / 255.0
    X = np.expand_dims(X, axis=-1)
    y = np.array(target)


    return X, y

def model_creating():
     
    X_train, y_train = train_data_load(train_dir)
    X_test, y_test = test_data_load(test_dir)


    model = tf.keras.Sequential([


        tf.keras.layers.Input(shape=(48, 48, 1)),

        tf.keras.layers.RandomFlip("horizontal"),  # Случайный переворот
        tf.keras.layers.RandomRotation(0.1),       # Поворот до 10%
        tf.keras.layers.RandomZoom(0.1),           # Зум до 10%
        #tf.keras.layers.RandomContrast(0.1),       # Контраст
        #tf.keras.layers.RandomTranslation(0.1, 0.1), # Сдвиг

            # 3. Первый сверточный слой. 32 фильтра размером 3x3.
        # activation='relu' — функция, которая убирает отрицательные числа (делает их 0), 
        # помогая сети учиться нелинейно.
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        
        # 4. Batch Normalization — "центрует" данные, чтобы нейроны не выдавали слишком большие числа. 
        # Это сильно ускоряет обучение.
        tf.keras.layers.BatchNormalization(),
        
        # 5. MaxPooling — сжимает картинку в 2 раза (станет 24x24). 
        # Оставляет только самые важные (яркие) пиксели.
        tf.keras.layers.MaxPooling2D((2, 2)),


        #Второй углубленный сверточный слой
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)), # Картинка сжимается до 12x12
        
        # 7. Dropout — случайно "выключает" 25% нейронов. 
        # Это заставляет сеть не зубрить картинку, а искать общие признаки.
        tf.keras.layers.Dropout(0.25),


            # 8. Третий слой свертки для еще более глубоких признаков.
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),


        tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),




        tf.keras.layers.MaxPooling2D((2, 2)), # Сжимаем до 6x6
        
        tf.keras.layers.Dropout(0.4),
        # 9. Flatten — "выпрямляет" матрицу 6x6x128 в один длинный ряд из 4608 чисел.
        tf.keras.layers.Flatten(),



            # 10. Dense (полносвязный) слой — 128 нейронов, которые "думают" над признаками.
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5), # Еще больше защиты от переобучения
        
        # 11. ФИНАЛЬНЫЙ СЛОЙ. 7 нейронов (по числу эмоций).
        # activation='softmax' превращает выходы в проценты (например, 0.8 счастья, 0.1 грусти).
        tf.keras.layers.Dense(7, activation='softmax')
        ])
    



    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True,
        verbose=1
    )

    # 2. Настройка сохранения модели
    # save_best_only=True — сохранять файл только если результат лучше предыдущего
    checkpoint = ModelCheckpoint(
        'best_emotion_model.keras', 
        monitor='val_accuracy', 
        save_best_only=True,
        verbose=1
    )


    model.compile(
    # Adam — самый популярный алгоритм для изменения весов нейронов (оптимизатор).
    optimizer='adam',
    
    # loss — функция ошибки. Sparse означает, что наши метки y — это просто числа (0, 1, 2...).
    loss='sparse_categorical_crossentropy',
    
    # Будем следить за точностью (сколько процентов угадано верно).
    metrics=['accuracy']
)

    model.summary()



    history = model.fit(
        X_train, y_train, 
        epochs=50, 
        batch_size=32,
        validation_data=(X_test, y_test), # Передаем твой x_test и y_test
        callbacks=[checkpoint] # Подключаем наши инструменты
    )
    
    return model, history
    


if __name__ == "__main__":

    model, history = model_creating()
        
        # Сохранение финальной модели (на всякий случай)
    model.save('D:/Projects/Face_emotion/models/final_emotion_model.keras')
    print("Обучение завершено и модель сохранена!")