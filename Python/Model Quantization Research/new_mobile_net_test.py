
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
import pandas as pd

'''
# Just an overview. Trained MobileNet on the MNIST dataset. 
This is a datset of hand drawn digits. Using mobilenet 
to classify these digits (10 classes for each of the digits).
I'm only focusing on post training quanitization. So we'll train 
mobile net on mnist and then quaniizes later. 
I'm following the link below really closely. it has all the code for quanticizing

https://www.tensorflow.org/model_optimization/guide/quantization/post_training

this has three types of post training quant: 
Post-training dynamic range quantization
Post-training full integer quantization
Post-training float16 quantization

I have code for all of them but i haven't run it yet on the full integer quant. 
This is all scratch!! going to clean it up and then save the graphs so we can present them. 
pretty primative stuff but we need something. 

her are other resources i used 

# https://www.tensorflow.org/model_optimization/guide/quantization/post_training
#https://ai.google.dev/edge/litert/models/post_training_quantization

- using tf lite which coverts model into quantizied form 


'''

# # # # load the MNIST dataset
# # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# # # # print the shape of the data
# # print("x_train shape:", x_train.shape)
# # print("y_train shape:", y_train.shape)
# # print("x_test shape:", x_test.shape)
# # print("y_test shape:", y_test.shape)

# # plt.imshow(x_train[0], cmap="gray")
# # # plt.show()

# # # now trying to use mobile net basic-- 

# # mnist data right now is 28x28.. but defautl for mobile net is 224x224x3 
# # 224x224 size with 3 channels (RGB)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


def preprocess(image, label):
    image = tf.expand_dims(image, -1)  # add channel dimension (28, 28, 1)
    image = tf.image.grayscale_to_rgb(image)  # convert grayscale to RGB (28, 28, 3)
    image = tf.image.resize(image, [224, 224])  # resize for MobileNet
    image = image / 255.0  # normalize -- each pixel val takes on value from 1-225, normalized so that all vals are between 0 and 1 
    return image, label

# create tensorFlow dataset pipeline-- perhaps not neccary but run out of RAM when i just try to load the images in...
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)

# # load MobileNetV2 without top layers
# # weights='imagenet': specifies that we are using the pre-trained weights of the model trained on the ImageNet dataset
# # include_top=False: indicates that we do not want to use the final dense layer of the ResNet model, which is responsible for classifying images into the original ImageNet categories
# # Since we are using the ResNet50 model for classifying images for our own task, we will include our own top layer.
# base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# # # prevents the weights from being updated at each layer
# # # want to leverage the knowledge captured from pre-trained model
# base_model.trainable = False

# # add custom classifier
# # adding global pooling layer to reduce spatial dimensions and provide global summary of the features
# # avg pooling computes the average of each feature map in the last conv layer
# x = GlobalAveragePooling2D()(base_model.output)
# x = Dense(128, activation="relu")(x)
# x = Dropout(0.5)(x)
# output = Dense(10, activation="softmax")(x)  # 10 output classes for MNIST

# # build and compile model
# model = Model(inputs=base_model.input, outputs=output)
# model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# # train the model
# # model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=32)
# model.fit(train_ds, validation_data=test_ds, epochs=5)

# # Evaluate on test data
# # test_loss, test_acc = model.evaluate(x_test, y_test)
# # print(f"Test Accuracy: {test_acc:.4f}")
# test_loss, test_acc = model.evaluate(test_ds)
# print(f"Test Raw Accuracy: {test_acc:.4f}")
# model.save("mobilenet_mnist.h5")


# model = load_model("mobilenet_mnist.h5")
# # getting classifcation report from this 
# y_true = np.concatenate([y.numpy() for _, y in test_ds])  # Extract true labels
# y_pred_probs = model.predict(test_ds)  # Get predicted probabilities
# y_pred = np.argmax(y_pred_probs, axis=1)  # Convert probabilities to class labels

# # Generate confusion matrix
# cm_base = confusion_matrix(y_true, y_pred)

# plt.figure(figsize=(8, 6))
# sns.heatmap(cm_base, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.title("Confusion Matrix")
# plt.show()

# Generate classification report
# report_base = classification_report(y_true, y_pred, digits=4)
# print(report_base)


model = load_model("mobilenet_mnist.h5")

# Extract true labels
y_true = np.concatenate([y.numpy() for _, y in test_ds])

# Get predictions
y_pred_probs = model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

# Generate classification report
report_base = classification_report(y_true, y_pred, digits=4, output_dict=True)

# Convert report to DataFrame
df_report = pd.DataFrame(report_base).transpose()

# Save report as CSV
df_report.to_csv("base_model_report.csv", index=True)

# Generate confusion matrix
cm_base = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_base, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


# import pandas as pd
# from sklearn.metrics import classification_report

# # Example classification report
# report_base = classification_report(y_true, y_pred, digits=4, output_dict=True)  # Convert to dict

# # Convert dict to DataFrame
# df_report_base = pd.DataFrame(report_base).transpose()

# # Print DataFrame
# print(df_report_base)


#### using tf lite which coverts model into quantizied form 
'''
# this is all post training quantization-- look into three types 
    # dynamic range quantization: 
            # quantizes only the weights from floating point to integer, provides 8bits of presicion
    # full integer quantization 
        # can either do with fallback or integer only 
       # fully integer quantize a model-- use float operators when they don't have 
       an integer implemnetaiton 
    # float 16 quantization
        # reduce size of a floating point model by quantizing the weights to flaot 16
# https://www.tensorflow.org/model_optimization/guide/quantization/post_training
#https://ai.google.dev/edge/litert/models/post_training_quantization

- using tf lite which coverts model into quantizied form 
- https://www.tensorflow.org/model_optimization/guide/quantization/post_training
so this would be post training quantization, since we've already trained our 
model on the mnist dataset
- this quantizes weights-- specifiies 8 bit integer weight quantization
'''

# Dynamic ramge quantization

# covert model to tf lite fromat with quantization
'''
# tf.lite.Optimize.DEFAULT: statically quanitzates only the weights 
# from floating point 
'''
# model = tf.keras.models.load_model("mobilenet_mnist.h5")
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT] 
# tflite_dynamic_quant_model = converter.convert()

# # save the quantized model
# with open("mobilenet_dynamic_quantized.tflite", "wb") as f:
#     f.write(tflite_dynamic_quant_model)

# # load the quantized TFLite model
# interpreter = tf.lite.Interpreter(model_path="mobilenet_dynamic_quantized.tflite")
# interpreter.allocate_tensors()


# def evaluate_tflite_model_with_metrics(interpreter, test_ds, input_dtype=np.float32):
#     # Get input and output tensor indices
#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()

#     y_true = []
#     y_pred = []

#     for images, labels in test_ds:
#         for i in range(images.shape[0]):
#             input_data = np.expand_dims(images[i], axis=0).astype(input_dtype)  # Convert to specified type

#             # Set input tensor
#             interpreter.set_tensor(input_details[0]['index'], input_data)
#             interpreter.invoke()  # Run inference

#             # Get output tensor
#             output_data = interpreter.get_tensor(output_details[0]['index'])

#             # Convert output to predicted label
#             predicted_label = np.argmax(output_data)

#             y_true.append(labels[i].numpy())  # Actual label
#             y_pred.append(predicted_label)   # Predicted label

#     # Compute confusion matrix
#     cm = confusion_matrix(y_true, y_pred)

#     # Generate classification report
#     report = classification_report(y_true, y_pred, digits=4,output_dict=True)

#     return cm, report






# # compute confusion matrix and classification report
# conf_matrix_dynamic_quant, class_report_dynamic_quant = evaluate_tflite_model_with_metrics(interpreter, test_ds,input_dtype=np.float32)

# print("Confusion Matrix:")
# print(conf_matrix_dynamic_quant)

# print("\nClassification Report:")
# print(class_report_dynamic_quant)

# df = pd.DataFrame(class_report_dynamic_quant).transpose()
# df.to_csv("dynamic_quant_model_report.csv", index=True)

# print("Report saved as dynamic_quant_model_report.csv.csv")



# # now using integer only quanization 

############

# this is specifically for the full integer quantization-- i will combine it into a better function later 
def evaluate_tflite_model_with_metrics_full(interpreter, test_ds, input_dtype=np.float32):
    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # input_details[0] is a dict describing input tensor of TFLite model (shape,dtype,ect)
    input_scale, input_zero_point = input_details[0]['quantization']

    y_true = []
    y_pred = []

    for images, labels in test_ds:
        for i in range(images.shape[0]):
            img = images[i].numpy()

            if input_dtype == np.int8:
                # Apply quantization to match training calibration
                img = img / input_scale + input_zero_point
                img = np.clip(np.round(img), -128, 127).astype(np.int8)
            else:
                img = img.astype(np.float32)

            input_data = np.expand_dims(img, axis=0)

            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            output_data = interpreter.get_tensor(output_details[0]['index'])
            predicted_label = np.argmax(output_data)

            y_true.append(labels[i].numpy())
            y_pred.append(predicted_label)

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4, output_dict=True)

    return cm, report

'''
for below code, reference 
https://ai.google.dev/edge/litert/models/post_training_integer_quant#convert_using_integer-only_quantization
'''

def representative_data_gen():
    for images, _ in train_ds.take(100):  # Use a small subset (e.g., 100 batches)
        for i in range(images.shape[0]):
            yield [np.expand_dims(images[i].numpy(), axis=0).astype(np.float32)]

model = tf.keras.models.load_model("mobilenet_mnist.h5")

# set up TTLiteCoverter
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# set up representative dataset
converter.representative_dataset = representative_data_gen

# ensure full integer quant (weights+ activations)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# set input/output to int8
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# convert

tflite_full_int_quant_model = converter.convert()

# Save the model
with open("mobilenet_mnist_full_int.tflite", "wb") as f:
    f.write(tflite_full_int_quant_model)


# evalutate qunaitzed model 
# load the tflite model 

interpreter = tf.lite.Interpreter(model_path="mobilenet_mnist_full_int.tflite")
interpreter.allocate_tensors()

# evaluate 
conf_matrix_full_int_quant, class_report_full_int_quant= evaluate_tflite_model_with_metrics_full(interpreter, test_ds, input_dtype=np.int8)

report_df_full = pd.DataFrame(class_report_full_int_quant).transpose()
report_df_full.to_csv("full_integer_model_report.csv")


############


# # now using float 16 quantization
# # with float 16 quanitzation 
# this wokrs 

# model = tf.keras.models.load_model("mobilenet_mnist.h5")

# # Set up TFLite Converter
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Enable optimizations
# converter.target_spec.supported_types = [tf.float16]  # Use Float16 for weights

# # Convert the model
# tflite_fp16_model = converter.convert()
# with open("mnist_model_quant_f16.tflite", "wb") as f:
#     f.write(tflite_fp16_model)


# interpreter_fp16 = tf.lite.Interpreter(model_path="mnist_model_quant_f16.tflite")
# interpreter_fp16.allocate_tensors()


# # Evaluate the model
# cm_fp16, report_fp16 = evaluate_tflite_model_with_metrics(interpreter_fp16, test_ds, input_dtype=np.float32)

# print("Float16 Model Performance:\n", report_fp16)

# df = pd.DataFrame(report_fp16).transpose()
# df.to_csv("float16_model_report.csv", index=True)

# print("Report saved as float16_model_report.csv")
