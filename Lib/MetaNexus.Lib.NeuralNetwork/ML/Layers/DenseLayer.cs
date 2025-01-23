using MetaNexus.Lib.NeuralNetwork.ML.Layers.Abstractions;
using MetaNexus.Lib.NeuralNetwork.Tensors;

public class DenseLayer : Layer
{
    // Конструктор слоя
    public DenseLayer(int inputSize, int size) : base(inputSize, size)
    {
    }

    public DenseLayer(int inputSize, int size, Tensor weights, Tensor biases) : base(inputSize, size, weights, biases)
    {
    }

    public DenseLayer(int inputSize, int size, ActivationFunc activationFunction, ActivationPrimeFunc activationPrimeFunction)
        : base(inputSize, size, activationFunction, activationPrimeFunction)
    {
    }

    public DenseLayer(int inputSize, int size, Tensor weights, Tensor biases, ActivationFunc activationFunction, ActivationPrimeFunc activationPrimeFunction)
        : base(inputSize, size, weights, biases, activationFunction, activationPrimeFunction)
    {
    }

    /// <summary>
    /// Метод для выполнения прямого прохода через полносвязный слой.
    /// Здесь вычисляется результат работы с весами и смещениями.
    /// </summary>
    /// <param name="input">Входной тензор для слоя.</param>
    /// <returns>Выходной тензор после прохождения через слой.</returns>
    public override Tensor Forward(Tensor input)
    {
        this.input = input; // Сохраняем вход для использования в Backward

        // Логирование входных данных
        Console.WriteLine("Прямой проход через слой (Forward):");
        Console.WriteLine("Входные данные: " + input.ToString());

        // Формируем выходной тензор с размерностью [batch_size, size], где size — количество нейронов
        Tensor output = new Tensor(new int[] { input.Shape[0], Size });

        // Процесс вычисления линейного выхода
        for (int i = 0; i < Size; i++)
        {
            float sum = 0.0f;

            // Суммируем входы, умножаем на веса для каждого нейрона
            for (int j = 0; j < InputSize; j++)
            {
                sum += input[0, j] * weights[j, i];  // Суммируем продукты входных значений и весов
            }

            // Логирование каждого шага вычислений
            Console.WriteLine($"Для нейрона {i}: сумма входных значений и весов = {sum}");

            // Логирование смещений
            Console.WriteLine($"Смещение для нейрона {i}: {biases[i]}");

            // Суммируем и добавляем смещение
            output[0, i] = sum + biases[i];
        }

        // Логирование выхода до активации
        Console.WriteLine("Выход до активации: " + string.Join(", ", output.ToString()));

        // Применяем функцию активации
        Tensor activatedOutput = ApplyActivation(output);

        // Логирование выхода после активации
        Console.WriteLine("Выход после активации: " + string.Join(", ", activatedOutput.ToString()));

        return activatedOutput;
    }
}
