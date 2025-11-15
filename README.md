# ML - HPX

Implementation of common ML algorithms from scratch in C++, utilisng HPX as the runtime and providing bindings for them in Python using Nanobind.

## Build Instructions
Pre-requisites: [Installing HPX](https://hpx-docs.stellar-group.org/latest/html/quickstart.html)

```sh
cd build
git clone https://github.com/vrnimje/ml-hpx.git

cd ml-hpx
pip install .
```

### Utilising the bindings

1. k-Means Clustering

    [Script](./python_tests/kmeans.py):

    ```sh
    $ python python_tests/kmeans.py
    Sklearn KMeans time: 0.0168 seconds
    Inertia: 543.3955
    ML-HPX KMeans time: 0.0024 seconds
    Inertia: 543.3952
    ```

2. Perceptron classifier

    [Script](./python_tests/perceptron.py):

    ```sh
    $ python python_tests/perceptron.py
    ML-HPX Perceptron accuracy: 1.0000
    ML-HPX Perceptron time: 0.0016 seconds
    Sklearn Perceptron accuracy: 0.9995
    Sklearn Perceptron time: 0.0028 seconds
    ```

3. SVM Classifier

    [Script](./python_tests/svc.py):

    ```sh
    $ python python_tests/svc.py
    ML-HPX SVC accuracy: 1.0000
    ML-HPX SVC time: 0.0220 seconds
    Sklearn SVC accuracy: 0.9970
    Sklearn SVC time: 0.1196 seconds
    ```

4. Neural Network

    [Script](./python_tests/nn.py):

    Compared against tensorflow_cpu on 11th Gen Intel® Core™ i5-11300H × 8

    ```sh
    $ python python_tests/nn.py
    HPX NN training time: 0.001217 sec
    HPX Predictions: [[0.06723812758653255], [0.9431358899595929], [0.9532029734896459], [0.050276559281559836]]
    TensorFlow training time: 19.081340 sec
    TF Predictions: [[0.5016673803329468], [0.9870261549949646], [0.5016673803329468], [0.010493488050997257]]
    ```

    [Script](./python_tests/nn_2.py):

    ```sh
    TensorFlow training time: 9.295418 sec
    TF Accuracy: 0.9668, Loss: 0.0836
    TF Predictions (first 10): [0.9885914325714111, 0.11424487084150314, 0.014540525153279305, 0.007555932737886906, 0.00012698175851255655, 0.9988582730293274, 0.09589959681034088, 0.9277084469795227, 0.9998912215232849, 0.12737293541431427]
    HPX NN training time: 1.882052 sec
    HPX NN Accuracy: 0.9780, Loss: 0.0180
    HPX Predictions (first 10): [[0.9999939742043228], [0.9991363289582976], [1.195073234278672e-06], [5.306765734111496e-07], [7.105306254994499e-08], [0.9999999200370617], [0.0014173977894342772], [0.9998889419155234], [0.9998085469481175], [0.001822480821572805]]
    ```

## TO-DO

1. Re-pivot to NNs only
2. Layers like Normalization, SoftMax
3. CNNs ??
