using MetaNexus.Lib.NeuralNetwork.Tensors;

namespace MetaNexus.Lib.NeuralNetwork.ML.Layers.Abstractions
{
    public abstract class Layer
    {
        public Tensor Biases { get; private set; }
        public Tensor Weights { get; private set; }
        public int Size { get; protected set; }
        public int InputSize { get; protected set; }

        private static Random random = new Random(DateTime.Now.Millisecond);

        public Layer(int inputSize, int size)
        {
            if (size <= 0)
                throw new ArgumentException("Размер слоя должен быть положительным числом.");

            Size = size;
            InputSize = inputSize;

            InitializeWeightsAndBiases();
        }

        public Layer(int inputSize)
        {
            if (inputSize <= 0)
                throw new ArgumentException("Размер слоя должен быть положительным числом.");

            Size = inputSize;
            InputSize = inputSize;

            InitializeWeightsAndBiases();
        }

        public abstract Tensor Forward(Tensor input);

        private void InitializeWeightsAndBiases()
        {
            Weights = new Tensor(new int[] { InputSize, Size });
            Biases = new Tensor(new int[] { Size });

            // Инициализация весов и смещений случайными значениями
            Console.WriteLine("Инициализация весов и смещений:");

            // Инициализация весов
            for (int i = 0; i < InputSize; i++)
            {
                for (int j = 0; j < Size; j++)
                {
                    Weights[i, j] = (float)(random.NextDouble() * 2.0 - 1.0);
                    Console.WriteLine($"Вес для (нейрон {i}, {j}): {Weights[i, j]}");
                }
            }

            // Инициализация смещений
            for (int i = 0; i < Size; i++)
            {
                Biases[i] = (float)(random.NextDouble() * 2.0 - 1.0);
                Console.WriteLine($"Смещение для нейрона {i}: {Biases[i]}");
            }
        }

        public void SaveWeights(BinaryWriter writer)
        {
            // Сохранение весов и смещений в бинарный файл
            for (int i = 0; i < Weights.Shape[0]; i++)
            {
                for (int j = 0; j < Weights.Shape[1]; j++)
                {
                    writer.Write(Weights[i, j]);
                }
            }

            for (int i = 0; i < Biases.Shape[0]; i++)
            {
                writer.Write(Biases[i]);
            }
        }

        public void LoadWeights(BinaryReader reader)
        {
            // Загрузка весов
            for (int i = 0; i < Weights.Shape[0]; i++)
            {
                for (int j = 0; j < Weights.Shape[1]; j++)
                {
                    Weights[i, j] = reader.ReadSingle();
                }
            }

            // Загрузка смещений
            for (int i = 0; i < Biases.Shape[0]; i++)
            {
                Biases[i] = reader.ReadSingle();
            }
        }
    }
}
