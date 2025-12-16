import os
import logging
import numpy as np
from PIL import Image
import io
import pickle
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Модель
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2

class WeatherClassifierBot:
    def __init__(self, model_path='best_weather_classifier_final.h5', info_path='class_info.pkl'):
        self.model = None
        self.class_names = None
        self.img_size = None
        try:
            self.load_model(model_path, info_path)
            logger.info("Модель успешно загружена")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            raise
    
    def create_cnn_model(self, input_shape, num_classes):
        model = keras.Sequential([
            keras.layers.Input(shape=input_shape),
            keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(2),
            keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(2),
            keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(2),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(num_classes, activation='softmax')
        ])
        return model
    
    def create_mobilenet_model(self, input_shape, num_classes):
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape,
            pooling='avg'
        )
        base_model.trainable = False
        model = keras.Sequential([
            base_model,
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(num_classes, activation='softmax')
        ])
        return model
    
    def create_efficientnet_model(self, input_shape, num_classes):
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape,
            pooling='avg'
        )
        base_model.trainable = False
        model = keras.Sequential([
            base_model,
            keras.layers.Dropout(0.3),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(num_classes, activation='softmax')
        ])
        return model
    
    def load_model(self, model_path, info_path):
        if os.path.exists(info_path):
            with open(info_path, 'rb') as f:
                class_info = pickle.load(f)
            self.class_names = class_info['class_names']
            self.img_size = tuple(class_info['img_size'])
            best_model_name = class_info.get('best_model', 'EfficientNetB0')
            logger.info(f"Информация загружена из {info_path}")
            logger.info(f"Лучшая модель в обучении: {best_model_name}")
        else:
            self.class_names = ['cloudy', 'foggy', 'rainy', 'sunny']
            self.img_size = (128, 128)
            best_model_name = 'EfficientNetB0'
            logger.warning(f"Файл {info_path} не найден, используем значения по умолчанию")
        input_shape = self.img_size + (3,)
        num_classes = len(self.class_names)
        if best_model_name == 'CNN':
            self.model = self.create_cnn_model(input_shape, num_classes)
        elif best_model_name == 'MobileNetV2':
            self.model = self.create_mobilenet_model(input_shape, num_classes)
        else:  # EfficientNetB0 или по умолчанию
            self.model = self.create_efficientnet_model(input_shape, num_classes)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        if os.path.exists(model_path):
            try:
                self.model.load_weights(model_path)
                logger.info(f"Веса модели загружены из {model_path}")
            except Exception as e:
                logger.warning(f"Не удалось загрузить веса из {model_path}: {e}")
                try:
                    self.model = keras.models.load_model(model_path, compile=False)
                    logger.info(f"Вся модель загружена из {model_path}")
                except Exception as e2:
                    logger.error(f"Не удалось загрузить модель: {e2}")
                    raise
        else:
            logger.warning(f"Файл модели {model_path} не найден, используем случайные веса")
        logger.info(f"Классы: {self.class_names}")
        logger.info(f"Размер изображений: {self.img_size}")
    
    def preprocess_image(self, image_bytes):
        try:
            img = Image.open(io.BytesIO(image_bytes))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize(self.img_size)
            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            return img_array, img
        except Exception as e:
            logger.error(f"Ошибка предобработки изображения: {e}")
            raise
    
    def predict(self, image_bytes):
        if self.model is None:
            return {"error": "Модель не загружена"}
        try:
            img_array, img = self.preprocess_image(image_bytes)
            predictions = self.model.predict(img_array, verbose=0)[0]
            top_indices = np.argsort(predictions)[-3:][::-1]
            top_predictions = [
                (self.class_names[i], float(predictions[i]))
                for i in top_indices
            ]
            predicted_idx = np.argmax(predictions)
            predicted_class = self.class_names[predicted_idx]
            confidence = float(predictions[predicted_idx])
            all_predictions = {
                cls: float(prob) 
                for cls, prob in zip(self.class_names, predictions)
            }
            return {
                'success': True,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'top_predictions': top_predictions,
                'all_predictions': all_predictions,
                'image': img
            }
            
        except Exception as e:
            logger.error(f"Ошибка предсказания: {e}")
            return {"error": str(e)}

# Бот
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    ConversationHandler
)

CHOOSING = 1

class TelegramWeatherBot:
    def __init__(self, token, classifier):
        self.token = token
        self.classifier = classifier
        self.application = None
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        
        emoji_map = {
            'sunny': '☀️',
            'cloudy': '☁️',
            'rainy': '🌧️',
            'foggy': '🌫️'
        }
        
        classes_list = "\n".join([
            f"• {emoji_map.get(cls, '📊')} {cls.capitalize()}"
            for cls in self.classifier.class_names
        ])
        
        welcome_text = (
            f"Привет, {user.first_name}!\n\n"
            "Я - бот для классификации погоды по фотографиям.\n\n"
            f"Я могу определить следующие типы погоды:\n{classes_list}\n\n"
            "Просто отправь мне фотографию, и я скажу, какая на ней погода!\n\n"
        )
        keyboard = [
            ["📸 Отправить фото"]
        ]
        reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
        await update.message.reply_text(welcome_text, reply_markup=reply_markup, parse_mode='Markdown')
        return CHOOSING
    
    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            processing_msg = await update.message.reply_text(
                "Анализирую изображение...",
                reply_to_message_id=update.message.message_id
            )
            
            photo_file = await update.message.photo[-1].get_file()
            photo_bytes = await photo_file.download_as_bytearray()
            
            result = self.classifier.predict(photo_bytes)
            
            if 'error' in result:
                await processing_msg.edit_text(f"❌ Ошибка: {result['error']}")
                return CHOOSING
            
            response = self.format_prediction(result)
            await processing_msg.edit_text(response, parse_mode='Markdown')
            
            logger.info(f"Предсказание для пользователя {update.effective_user.id}: {result['predicted_class']} ({result['confidence']:.2%})")
            
        except Exception as e:
            logger.error(f"Ошибка обработки фото: {e}")
            error_msg = (
                "Произошла ошибка при обработке фото\n\n"
            )
            await update.message.reply_text(error_msg, parse_mode='Markdown')
        
        return CHOOSING
    
    def format_prediction(self, result):
        emoji_map = {
            'sunny': '☀️',
            'cloudy': '☁️',
            'rainy': '🌧️',
            'foggy': '🌫️'
        }
        
        pred_class = result['predicted_class']
        confidence = result['confidence']
        top_preds = result['top_predictions']
        
        if confidence > 0.8:
            status = "Высокая уверенность"
            status_emoji = "✅"
        elif confidence > 0.6:
            status = "Средняя уверенность"
            status_emoji = "⚠️"
        else:
            status = "Низкая уверенность"
            status_emoji = "❓"
        
        emoji = emoji_map.get(pred_class.lower(), '📊')
        response = f"{emoji} Основное предсказание: {pred_class.capitalize()}\n"
        response += f"Уверенность: {confidence:.1%}\n"
        response += f"{status_emoji} {status}\n\n"
        
        response += "Топ-3 предсказания:\n"
        for i, (cls, prob) in enumerate(top_preds, 1):
            cls_emoji = emoji_map.get(cls.lower(), '📊')
            response += f"{cls_emoji} {cls.capitalize()}: {prob:.1%}\n"
        
        response += f"\nВсе вероятности:\n"
        for cls, prob in result['all_predictions'].items():
            cls_emoji = emoji_map.get(cls.lower(), '📊')
            bar_length = int(prob * 15)
            response += f"{cls_emoji} {cls.capitalize()}: {prob:.1%}\n"
        
        return response
    
    async def cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        keyboard = [["📸 Отправить фото", "ℹ️ Помощь"]]
        reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
        
        await update.message.reply_text(
            "Действие отменено.\n\n"
            "Используйте /start чтобы начать заново.",
            reply_markup=reply_markup
        )
        return ConversationHandler.END
    
    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        logger.error(f"Ошибка в обновлении {update}: {context.error}")
        try:
            if update and update.message:
                await update.message.reply_text(
                    "Произошла внутренняя ошибка бота.\n"
                    "Пожалуйста, попробуйте еще раз или используйте /start"
                )
        except:
            pass
    
    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        text = update.message.text
        if text == "📸 Отправить фото":
            await update.message.reply_text(
                "Отлично!\n\n"
                "Просто сделайте фото или выберите из галереи.\n\n"
            )
        else:
            await update.message.reply_text(
                "Я понимаю команды и фото!\n\n"
            )
        
        return CHOOSING
    
    def run(self):
        self.application = Application.builder().token(self.token).build()
        conv_handler = ConversationHandler(
            entry_points=[CommandHandler('start', self.start)],
            states={
                CHOOSING: [
                    CommandHandler('cancel', self.cancel),
                    MessageHandler(filters.PHOTO, self.handle_photo),
                    MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text),
                ],
            },
            fallbacks=[CommandHandler('cancel', self.cancel)],
        )
        
        self.application.add_handler(conv_handler)
        self.application.add_error_handler(self.error_handler)
        
        logger.info("Бот запущен...")
        print(f"Классы: {self.classifier.class_names}")
        print(f"Размер изображения: {self.classifier.img_size}")
        self.application.run_polling(allowed_updates=Update.ALL_TYPES)

def main():
    BOT_TOKEN = "$bot-token"
    model_files = ['best_weather_classifier_final.h5']
    missing_files = [f for f in model_files if not os.path.exists(f)]
    try:
        print("\nЗагружаем модель классификации...")
        classifier = WeatherClassifierBot()
        print("Запускаем телеграм-бота...")
        bot = TelegramWeatherBot(BOT_TOKEN, classifier)
        bot.run()
    except Exception as e:
        logger.error(f"Ошибка запуска бота: {e}")
        print(f"\n Ошибка: {e}")

if __name__ == "__main__":
    main()
