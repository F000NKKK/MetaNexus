using MetaNexus.Lib.NeuralNetwork.Tensors;
using MetaNexus.Lib.NeuralNetwork.Tensors.Abstractions;

namespace MetaNexus.Lib.NeuralNetwork.ML.Layers.Abstractions
{
    /// <summary>
    /// Абстрактный класс для слоя нейронной сети.
    /// </summary>
    public abstract class Layer : ILayer
    {
        /// <summary>
        /// Конструктор для создания слоя с размером входа равным размеру слоя,  с функцией активации по умолчанию (ReLU).
        /// </summary>
        /// <param name="inputSize">Размер входа.</param>
        public Layer(int inputSize)
            : this(inputSize, DefaultActivation, DefaultActivationPrime)
        {
            if (inputSize <= 0)
                throw new ArgumentException("Размер слоя должен быть положительным числом.");

            Size = inputSize;
            InputSize = inputSize;

            ((ILayer)this).InitializeWeightsAndBiases();
        }

        /// <summary>
        /// Конструктор для создания слоя с функцией активации по умолчанию (ReLU).
        /// </summary>
        /// <param name="inputSize">Размер входа.</param>
        /// <param name="size">Размер слоя (количество нейронов).</param>
        public Layer(int inputSize, int size)
            : this(inputSize, size, DefaultActivation, DefaultActivationPrime)
        {
            if (size <= 0)
                throw new ArgumentException("Размер слоя должен быть положительным числом.");

            ((ILayer)this).InitializeWeightsAndBiases();
        }

        /// <summary>
        /// Конструктор для создания слоя с заданными весами и смещениями,  с функцией активации по умолчанию (ReLU).
        /// </summary>
        /// <param name="inputSize">Размер входа.</param>
        /// <param name="size">Размер слоя (количество нейронов).</param>
        /// <param name="weights">Тензор весов для слоя.</param>
        /// <param name="biases">Тензор смещений для слоя.</param>
        public Layer(int inputSize, int size, Tensor weights, Tensor biases)
            : this(inputSize, size, weights, biases, DefaultActivation, DefaultActivationPrime)
        {
            if (weights.Shape[0] != inputSize || weights.Shape[1] != size)
                throw new ArgumentException("Размеры тензоров весов не совпадают с размером слоя.");

            if (biases.Shape[0] != size)
                throw new ArgumentException("Размеры тензора смещений не совпадают с размером слоя.");

            Size = size;
            InputSize = inputSize;
            this.weights = weights;
            this.biases = biases;
        }

        /// <summary>
        /// Конструктор для создания слоя с размером входа равным размеру слоя,  с функцией активации по умолчанию (ReLU).
        /// </summary>
        /// <param name="inputSize">Размер входа.</param>
        public Layer(
            int inputSize,
            ActivationFunc activationFunction,
            ActivationPrimeFunc activationPrimeFunction)
        {
            if (inputSize <= 0)
                throw new ArgumentException("Размер слоя должен быть положительным числом.");

            Size = inputSize;
            InputSize = inputSize;

            ActivationFunction = activationFunction ?? throw new ArgumentNullException(nameof(activationFunction));
            ActivationPrimeFunction = activationPrimeFunction ?? throw new ArgumentNullException(nameof(activationPrimeFunction));

            ((ILayer)this).InitializeWeightsAndBiases();
        }

        /// <summary>
        /// Конструктор для создания слоя с заданными размерами и функцией активации.
        /// </summary>
        /// <param name="inputSize">Размер входа.</param>
        /// <param name="size">Размер слоя (количество нейронов).</param>
        /// <param name="activationFunction">Функция активации.</param>
        /// <param name="activationPrimeFunction">Производная функции активации.</param>
        public Layer(
            int inputSize,
            int size,
            ActivationFunc activationFunction,
            ActivationPrimeFunc activationPrimeFunction)
        {
            if (size <= 0)
                throw new ArgumentException("Размер слоя должен быть положительным числом.");

            Size = size;
            InputSize = inputSize;

            ActivationFunction = activationFunction ?? throw new ArgumentNullException(nameof(activationFunction));
            ActivationPrimeFunction = activationPrimeFunction ?? throw new ArgumentNullException(nameof(activationPrimeFunction));

            ((ILayer)this).InitializeWeightsAndBiases();
        }

        /// <summary>
        /// Конструктор для создания слоя с заданными весами, смещениями и функцией активации.
        /// </summary>
        /// <param name="inputSize">Размер входа.</param>
        /// <param name="size">Размер слоя (количество нейронов).</param>
        /// <param name="weights">Тензор весов для слоя.</param>
        /// <param name="biases">Тензор смещений для слоя.</param>
        /// <param name="activationFunction">Функция активации.</param>
        /// <param name="activationPrimeFunction">Производная функции активации.</param>
        public Layer(
            int inputSize,
            int size,
            Tensor weights,
            Tensor biases,
            ActivationFunc activationFunction,
            ActivationPrimeFunc activationPrimeFunction)
        {
            if (weights.Shape[0] != inputSize || weights.Shape[1] != size)
                throw new ArgumentException("Размеры тензоров весов не совпадают с размером слоя.");

            if (biases.Shape[0] != size)
                throw new ArgumentException("Размеры тензора смещений не совпадают с размером слоя.");

            Size = size;
            InputSize = inputSize;

            this.weights = weights;
            this.biases = biases;

            ActivationFunction = activationFunction ?? throw new ArgumentNullException(nameof(activationFunction));
            ActivationPrimeFunction = activationPrimeFunction ?? throw new ArgumentNullException(nameof(activationPrimeFunction));
        }

        void ILayer.InitializeWeightsAndBiases()
        {
            weights = new Tensor(new int[] { InputSize, Size });
            biases = new Tensor(new int[] { Size });

            for (int i = 0; i < InputSize; i++)
            {
                for (int j = 0; j < Size; j++)
                {
                    weights[i, j] = (float)(random.NextDouble() * 2.0 - 1.0);
                }
            }

            for (int i = 0; i < Size; i++)
            {
                biases[i] = (float)(random.NextDouble() * 2.0 - 1.0);
            }
        }

        public Tensor GetWeights() => weights;

        public void SetWeights(Tensor newWeights)
        {
            if (newWeights.Shape[0] != InputSize || newWeights.Shape[1] != Size)
                throw new ArgumentException("Размеры нового тензора весов не совпадают с размерами слоя.");

            weights = newWeights;
        }

        public Tensor GetBiases() => biases;

        public void SetBiases(Tensor newBiases)
        {
            if (newBiases.Shape[0] != Size)
                throw new ArgumentException("Размеры нового тензора смещений не совпадают с размерами слоя.");

            biases = newBiases;
        }

        public abstract Tensor Forward(Tensor input);

        public Tensor Backward(Tensor delta, float learningRate)
        {
            if (!(!Equals(input, null) && input.Shape != null))
            {
                throw new ArgumentNullException(nameof(input), "Входной тензор не может быть null.");
            }

            // 1. Градиенты для весов и смещений
            Tensor weightGradient = input.Transpose().Dot(delta); // dE/dW - градиент весов
            Tensor biasGradient = delta.Sum(axis: 0);              // dE/db - градиент смещений

            // Приводим biasGradient к правильной размерности, если необходимо
            if (biasGradient.Shape[0] == 1 && biasGradient.Shape[1] == biases.Shape[0])
            {
                biasGradient = biasGradient.Flatten();  // Приводим к одномерному тензору
            }

            // 2. Обновление весов и смещений
            weights -= weightGradient * learningRate;             // W = W - η * dE/dW  
            biases -= biasGradient * learningRate;                // b = b - η * dE/db 

            var clipSize = 5.0f;

            // Ограничение максимального значения градиентов
            weightGradient = weightGradient.Clip(-clipSize, clipSize);
            biasGradient = biasGradient.Clip(-clipSize, clipSize);

            // 3. Градиенты для предыдущего слоя
            Tensor previousDelta = delta.Dot(weights.Transpose()); // dE/dInput

            return previousDelta;
        }



        /// <summary>
        /// Тензор смещений для слоя.
        /// </summary>
        public Tensor biases;

        /// <summary>
        /// Тензор весов для слоя.
        /// </summary>
        public Tensor weights;

        /// <summary>
        /// Тензор ввода для обратного хода
        /// </summary>
        public Tensor input;

        /// <summary>
        /// Делегат, представляющий функцию активации.
        /// </summary>
        /// <param name="input">Входной тензор.</param>
        /// <returns>Результат применения функции активации.</returns>
        public delegate Tensor ActivationFunc(ITensor input);

        /// <summary>
        /// Делегат, представляющий производную функции активации.
        /// </summary>
        /// <param name="input">Входной тензор.</param>
        /// <returns>Результат применения производной функции активации.</returns>
        public delegate Tensor ActivationPrimeFunc(ITensor input);

        /// <summary>
        /// Функция активации для слоя.
        /// </summary>
        public ActivationFunc ActivationFunction { get; set; }

        /// <summary>
        /// Производная функции активации для слоя.
        /// </summary>
        public ActivationPrimeFunc ActivationPrimeFunction { get; set; }

        public Tensor ApplyActivation(Tensor input)
        {
            if (ActivationFunction == null)
                throw new InvalidOperationException("Функция активации не установлена.");
            return ActivationFunction(input);
        }

        public Tensor ApplyActivationPrime(Tensor input)
        {
            if (ActivationPrimeFunction == null)
                throw new InvalidOperationException("Производная функции активации не установлена.");
            return ActivationPrimeFunction(input);
        }

        /// <summary>
        /// Функция активации по умолчанию (ReLU).
        /// </summary>
        private static Tensor DefaultActivation(ITensor input) => ((Tensors.Abstractions.ITensorActivationOperations)input).ApplyReLU();

        /// <summary>
        /// Производная функции активации по умолчанию (ReLU').
        /// </summary>
        private static Tensor DefaultActivationPrime(ITensor input) => ((Tensors.Abstractions.ITensorActivationOperationsPrime)input).ApplyReLUPrime();


        /// <summary>
        /// Размер слоя (количество нейронов в слое).
        /// </summary>
        public int Size { get; set; }

        /// <summary>
        /// Размер входа (количество нейронов или признаков на входе).
        /// </summary>
        public int InputSize { get; set; }

        /// <summary>
        /// Случайный генератор для инициализации весов и смещений.
        /// </summary>
        private static Random random = new Random(DateTime.Now.Millisecond);
    }
}
