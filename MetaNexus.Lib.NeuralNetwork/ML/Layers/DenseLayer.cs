using MetaNexus.Lib.NeuralNetwork.Math.Functions;
using MetaNexus.Lib.NeuralNetwork.ML.Layers.Abstractions;
using System;
using System.Linq;

public class DenseLayer : Layer
{
    public string Activation { get; set; }

    /// <summary>
    /// Конструктор для полносвязного слоя.
    /// </summary>
    /// <param name="inputSize">Количество входных нейронов.</param>
    /// <param name="size">Количество нейронов в слое.</param>
    /// <param name="activation">Функция активации для слоя.</param>
    public DenseLayer(int inputSize, int size, string activation) : base(inputSize, size)
    {
        Activation = activation;
        Console.WriteLine($"Создан слой с {inputSize} входами и {size} нейронами. Активируем с функцией: {activation}");
    }

    /// <summary>
    /// Метод для выполнения прямого прохода через полносвязный слой.
    /// Здесь вычисляется результат работы с весами и смещениями.
    /// </summary>
    /// <param name="input">Входные данные для слоя.</param>
    /// <returns>Выходные данные после прохождения через слой.</returns>
    public override float[] Forward(float[] input)
    {
        // Применение нормализации (если необходимо)
        input = Normalize(input);

        Console.WriteLine("Прямой проход через слой (Forward):");

        // Логирование входных данных
        Console.WriteLine("Входные данные: " + string.Join(", ", input));

        // Вычисление выхода слоя
        var output = Enumerable.Range(0, Size)
                               .Select(i =>
                               {
                                   var sum = Enumerable.Range(0, InputSize)
                                                    .Select(j => input[j] * Weights[j, i])
                                                    .Sum();

                                   // Логирование каждого шага вычислений
                                   Console.WriteLine($"Для нейрона {i}: сумма входных значений и весов = {sum}");

                                   // Логирование смещений
                                   Console.WriteLine($"Смещение для нейрона {i}: {Biases[i]}");

                                   // Сумма и смещение
                                   return sum + Biases[i];
                               })
                               .ToArray();

        // Логирование выхода до активации
        Console.WriteLine("Выход до активации: " + string.Join(", ", output));

        // Применение функции активации
        var activatedOutput = ApplyActivation(output);

        // Логирование выхода после активации
        Console.WriteLine("Выход после активации: " + string.Join(", ", activatedOutput));

        return activatedOutput;
    }

    private float[] ApplyActivation(float[] input)
    {
        // Применение функции активации (например, Sigmoid) с использованием LINQ
        Console.WriteLine("Применяем функцию активации...");
        return input.Select(x => ActivationFunctions.Sigmoid(x)).ToArray();
    }
}