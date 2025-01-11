namespace MetaNexus.Lib.NeuralNetwork.ML.Layers.Abstractions
{
    public abstract class Layer
    {
        public float[] Biases { get; private set; }
        public float[,] Weights { get; private set; }
        public int Size { get; protected set; }
        public int InputSize { get; protected set; }

        public Layer(int inputSize, int size)
        {
            if (size <= 0)
                throw new ArgumentException("Размер слоя должен быть положительным числом.");

            Size = size;
            InputSize = inputSize;

            InitializeWeights();
        }

        public Layer(int inputSize)
        {
            if (inputSize <= 0)
                throw new ArgumentException("Размер слоя должен быть положительным числом.");

            Size = inputSize;
            InputSize = inputSize;

            InitializeWeights();
        }

        public abstract float[] Forward(float[] input);

        private void InitializeWeights()
        {
            Random random = new Random(DateTime.Now.Millisecond);
            Weights = new float[InputSize, Size];
            Console.WriteLine("Инициализация весов:");

            for (int i = 0; i < InputSize; i++)
            {
                for (int j = 0; j < Size; j++)
                {
                    Weights[i, j] = (float)(random.NextDouble() * 2.0 - 1.0);
                    // Логирование значений весов
                    Console.WriteLine($"Вес для (нейрон {i}, {j}): {Weights[i, j]}");
                }
            }

            // Инициализация смещений
            Biases = Enumerable.Range(0, Size)
                               .Select(_ => (float)(random.NextDouble() * 2.0 - 1.0))
                               .ToArray();

            // Логирование значений смещений
            Console.WriteLine("Инициализация смещений:");
            foreach (var bias in Biases)
            {
                Console.WriteLine($"Смещение: {bias}");
            }
        }

        public void SaveWeights(BinaryWriter writer)
        {
            foreach (var weight in Weights)
            {
                writer.Write(weight);
            }

            Biases.ToList().ForEach(bias => writer.Write(bias));
        }

        public void LoadWeights(BinaryReader reader)
        {
            for (int i = 0; i < InputSize; i++)
            {
                for (int j = 0; j < Size; j++)
                {
                    Weights[i, j] = reader.ReadSingle();
                }
            }

            Biases = Enumerable.Range(0, Size)
                               .Select(_ => reader.ReadSingle())
                               .ToArray();
        }

        // Нормализация функции
        protected float[] Normalize(float[] input)
        {
            float mean = input.Average();
            float variance = input.Select(val => (val - mean) * (val - mean)).Average();
            float stddev = MathF.Sqrt(variance);

            float[] normalized = input.Select(x => (x - mean) / (stddev + 1e-8f)).ToArray();
            return normalized;
        }
    }
}