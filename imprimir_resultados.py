import os
import matplotlib.pyplot as plt

# Crear la carpeta de salida si no existe
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Extraer los datos del historial de entrenamiento
accR2 = history.history['r_squared']
val_accR2 = history.history['val_r_squared']
accMAPE = history.history['mape']
val_accMAPE = history.history['val_mape']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(accR2))

# Graficar R² y pérdida
plt.figure(figsize=(12, 8))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, accR2, label='Training R²')
plt.plot(epochs_range, val_accR2, label='Validation R²')
plt.legend(loc='lower right')
plt.title('Training and Validation R²')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss (MAE)')
plt.plot(epochs_range, val_loss, label='Validation Loss (MAE)')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss (MAE)')

# Guardar la gráfica
plt.savefig(os.path.join(output_dir, 'r_squared_and_loss.png'))
plt.close()

# Graficar MAPE y pérdida
plt.figure(figsize=(12, 8))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, accMAPE, label='Training MAPE')
plt.plot(epochs_range, val_accMAPE, label='Validation MAPE')
plt.legend(loc='lower right')
plt.title('Training and Validation MAPE')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss (MAE)')
plt.plot(epochs_range, val_loss, label='Validation Loss (MAE)')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss (MAE)')

# Guardar la gráfica
plt.savefig(os.path.join(output_dir, 'mape_and_loss.png'))
plt.close()


#%% Valido el modelo norm Vs original

data_test = df[df['Ano'] == 2021]
Ener_test       = data_test['Demanda_ener'].unique()[0]
custom_objects  = {'r_squared': r_squared}
# labels_real     = df['Demanda']

model.save('modelos/UTE_ANN_V2'+tipo+str(EPOCHS))

tipo = 'norm'
data_test, label_test, coumnas_eliminadas, features_size = transform_data_input(data_test, version, iteracion, tipo)
model_ANN_norm = load_model('modelos/UTE_ANN_V2'+tipo+str(EPOCHS), custom_objects=custom_objects)
loss1, acc1, mape1  = model_ANN_norm.evaluate(data_test, label_test)
pred_norm  = model_ANN_norm.predict(data_test) * Ener_test/ Ener_2021

tipo = 'OneHot'
data_test, label_test, coumnas_eliminadas, features_size = transform_train(data_test, version, iteracion, tipo)
model_ANN_oneHot = load_model('modelos/UTE_ANN_V2'+tipo , custom_objects=custom_objects)
loss2, acc2, mape2 = model_ANN_oneHot.evaluate(data_test, label_test)
pred_oneHot  = model_ANN_oneHot.predict(data_test) * Ener_test/ Ener_2021
 
tipo = ''
data_test, label_test, coumnas_eliminadas, features_size = transform_train(data_test, version, iteracion, tipo)
model_ANN = load_model('modelos/UTE_ANN_V2'+tipo , custom_objects=custom_objects)
loss3, acc3, mape3 = model_ANN.evaluate(data_test, label_test)
pred_ANN  = model_ANN.predict(data_test)


#%%
ini = 180
timeslots = 168 
plt.plot(range(timeslots), pred_norm[ini: ini+timeslots],   label= 'Predict_norm')
plt.plot(range(timeslots), pred_oneHot[ini: ini+timeslots], label= 'Predict_oneHot')
plt.plot(range(timeslots), pred_ANN[ini: ini+timeslots],    label= 'Predict_tradic')
plt.plot(range(timeslots), label_val[ini: ini+timeslots],   label= 'Real')
plt.legend(loc='upper right')
plt.title('Predicción_norm Vs Predict_baU Vs Real')
plt.show()

# print("acc train_history: acc= {}, loss= {}".format(history.history['val_r_squared'][-1],history.history['val_loss'][-1]))
print("modelo norm   validacion: acc= {}, mape= {}, loss= {}".format(acc1, loss1, mape1))
print("modelo oneHot validacion: acc= {}, mape= {}, loss= {}".format(acc2, loss2, mape2))
print("modelo tradic validacion: acc= {}, mape= {}, loss= {}".format(acc3, loss3, mape3))
