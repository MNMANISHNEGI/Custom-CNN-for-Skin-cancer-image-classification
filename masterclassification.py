import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LeakyReLU, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import random
import os
import pandas as pd
import seaborn as sns


# Set the random seed for reproducibility
rnd_seed = 123
random.seed(rnd_seed)

# Define the directory paths (adjust these paths)
data_dir_train = 'C:\\Users\\91789\\Desktop\\Skin cancer ISIC The International Skin Imaging Collaboration\\train'
data_dir_test = 'C:\\Users\\91789\\Desktop\\Skin cancer ISIC The International Skin Imaging Collaboration\\test'

# Define image size and batch size
img_height = 180
img_width = 180
batch_size = 32
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir_train,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir_train,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir_test,
  validation_split=0.9,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# Load training dataset with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=40,
    horizontal_flip=True,
    validation_split=0.2
)
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
# Load training and validation sets
train_generator = train_datagen.flow_from_directory(
    data_dir_train,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    data_dir_train,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Load test dataset without data augmentation
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    data_dir_test,
    seed=rnd_seed,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

# Obtain class names correctly
class_names = list(train_generator.class_indices.keys())
print(class_names)
num_classes = len(class_names)

total = 0
all_count = []
class_name = []

for i in range(num_classes):
    class_dir = os.path.join(data_dir_train, class_names[i])
    count = len(os.listdir(class_dir))
    total += count
    print("Class name = ", class_names[i])
    print("count      = ", count)
    print("proportion = ", count/total)
    print("-------------------------------------")
    all_count.append(count)
    class_name.append(class_names[i])
#count all
temp_df = pd.DataFrame(list(zip(all_count, class_name)), columns=['count', 'class_name'])
sns.barplot(data=temp_df, y="count", x="class_name")
plt.xticks(rotation=90)
plt.show()

path_to_training_dataset = 'C:\\Users\\91789\\Desktop\\Skin cancer ISIC The International Skin Imaging Collaboration\\train'
import Augmentor


# Iterate over class names
for i in class_names:
    # Specify the path to the source directory containing images
    source_dir = os.path.join(path_to_training_dataset, i)

    # Set the random seed for reproducibility
    rnd_seed = 123
    random.seed(rnd_seed)
    # Specify the output directory for augmented images
    output_dir = os.path.join(path_to_training_dataset, i, 'output')

    # Create an Augmentor pipeline and add the source directory
    p = Augmentor.Pipeline(source_directory=source_dir, output_directory=output_dir)

    # Add augmentation operations
    p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)

    # Sample 1000 augmented images
    p.sample(1000)

num_classes = len(class_names)
total = 0
all_count = []
class_name = []


for i in range(num_classes):
    count = len(os.listdir(os.path.join(path_to_training_dataset, class_names[i], 'output')))
    total += count
    print("Class name = ", class_names[i])
    print("count      = ", count)
    print("proportion = ", count / total)
    print("-------------------------------------")
    all_count.append(count)
    class_name.append(class_names[i])


temp_df = pd.DataFrame(list(zip(all_count, class_name)), columns = ['count', 'class_name'])
sns.barplot(data=temp_df, y="count", x="class_name")
plt.xticks(rotation=90)
plt.show()


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  output_dir,
  seed=123,
  validation_split = 0.2,
  subset = 'training',
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  output_dir,
  seed=123,
  validation_split = 0.2,
  subset = 'validation',
  image_size=(img_height, img_width),
  batch_size=batch_size)
# Custom CNN model
model = Sequential()

# Layer 1
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(img_height, img_width, 3), padding='same'))
model.add(LeakyReLU(alpha=0.01))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

# Layer 2
model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
model.add(LeakyReLU(alpha=0.01))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

# Layer 3
model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
model.add(LeakyReLU(alpha=0.01))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

# Layer 4
model.add(Conv2D(128, kernel_size=(5, 5), padding='same'))
model.add(LeakyReLU(alpha=0.01))

# Layer 5
model.add(Conv2D(128, kernel_size=(5, 5), padding='same'))
model.add(LeakyReLU(alpha=0.01))

# Flatten the output before dense layers
model.add(Flatten())

# Dense Neural Network
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

# Output layer with Softmax activation for 9 classes
model.add(Dense(9, activation='softmax'))

# Compile the model
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()

# Fit the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=25,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator)
)

# Save the model
model.save('custom_skin_cancer_model_finalreview.h5')

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
