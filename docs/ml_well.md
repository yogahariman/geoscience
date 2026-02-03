
saya mau prediksi classification seperti lithology, regression seperti porosity, dan clustering di well dg python
struktur project :
geoscience/
├── geosc/
│   ├── seismic/
│   ├── well/

contoh dibawah saya ambilkan dari matlab

untuk training saya perlu :
tentukan nilai null value dari data
arsitektur NN dari sklearn.neural_network

fungsi yang dibuat
well log porosity training
well log porosity predicting

berikut contoh langkah2 sebelum memanggil fungsi training
nullVal=-999.25;
epochs=10000;
learning_rate = 0.001;
goal=0.001;
neuron=[5 5 5 1];
lyrActv=[tansig tansig tansig sigmoid];

T=readtable('data_training.csv');



fungsi train porosity

data training di preprocessing dulu

data = [T.GR T.NPHI];
ida = true(size(data,1),1);
for ii=1:size(data,2)
    ida = ida & data(:,ii) ~= nullVal;
end
 
data = data(ida,:);

MLPRegressor(
        hidden_layer_sizes=(hidden_neurons,),
        learning_rate_init=learning_rate,
        activation=activation,
        solver="adam",        # bisa diganti sgd kalau mau pure BP
        max_iter=max_epoch,
        batch_size=batch_size,
        random_state=42
    )

save knowledge

berikut contoh langkah2 sebelum memanggil fungsi predicting
nullVal = -999.25;
T=readtable('data_predicting.csv');
 
data = [T.GR T.NPHI];
 
