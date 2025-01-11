namespace MetaNexus.Lib.NeuralNetwork.ML.Layers.Abstractions
{
    /// <summary>
    /// Абстрактный класс для слоя нейронной сети.
    /// </summary>
    public abstract class Layer
    {
        /// <summary>
        /// Смещения (bias) для каждого нейрона в слое.
        /// </summary>
        public float[] Biases { get; private set; }

        /// <summary>
        /// Веса для каждого нейрона в слое.
        /// </summary>
        public float[] Weights { get; private set; }

        /// <summary>
        /// Количество нейронов в слое.
        /// </summary>
        public int Size { get; protected set; }

        /// <summary>
        /// Конструктор для слоя.
        /// </summary>
        /// <param name="size">Количество нейронов в слое. Значение должно быть больше нуля.</param>
        /// <exception cref="ArgumentException">Если размер слоя меньше или равен нулю.</exception>
        public Layer(int size)
        {
            if (size <= 0)
                throw new ArgumentException("Размер слоя должен быть положительным числом.");

            Size = size;
            InitializeWeights(); // Инициализация весов и смещений
        }

        /// <summary>
        /// Метод для выполнения прямого прохода через слой.
        /// </summary>
        /// <param name="input">Входные данные для слоя.</param>
        /// <returns>Выходные данные после прохождения через слой.</returns>
        public abstract float[] Forward(float[] input);

        /// <summary>
        /// Инициализация весов и смещений случайными значениями.
        /// </summary>
        private void InitializeWeights()
        {
            Random random = new Random();
            Weights = new float[Size];  // Пример: каждый нейрон имеет свой вес
            Biases = new float[Size];   // Смещения для каждого нейрона

            for (int i = 0; i < Size; i++)
            {
                Weights[i] = (float)(random.NextDouble() * 2.0 - 1.0);  // Инициализация случайными числами от -1 до 1
                Biases[i] = 0f;  // Инициализация смещений в ноль
            }
        }

        /// <summary>
        /// Сохранение весов и смещений слоя в бинарный файл.
        /// </summary>
        /// <param name="fs">Поток для записи в файл.</param>
        public void SaveWeights(FileStream fs)
        {
            using (var writer = new BinaryWriter(fs))
            {
                // Сохраняем веса
                foreach (var weight in Weights)
                {
                    writer.Write(weight);
                }

                // Сохраняем смещения
                foreach (var bias in Biases)
                {
                    writer.Write(bias);
                }
            }
        }

        /// <summary>
        /// Загрузка весов и смещений слоя из бинарного файла.
        /// </summary>
        /// <param name="fs">Поток для чтения из файла.</param>
        public void LoadWeights(FileStream fs)
        {
            using (var reader = new BinaryReader(fs))
            {
                // Загружаем веса
                Weights = new float[Size];
                for (int i = 0; i < Weights.Length; i++)
                {
                    Weights[i] = reader.ReadSingle();
                }

                // Загружаем смещения
                Biases = new float[Size];
                for (int i = 0; i < Biases.Length; i++)
                {
                    Biases[i] = reader.ReadSingle();
                }
            }
        }
    }
}
