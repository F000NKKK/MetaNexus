using MetaNexus.Lib.NeuralNetwork.ML.Layers.Abstractions;
using MetaNexus.Lib.NeuralNetwork.Tensors;
using System;
using System.Drawing;

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
        weights = new Tensor(new int[] { inputSize, size }); // Инициализация весов
        biases = new Tensor(new int[] { 1, size }); // Инициализация смещений

        ((ILayer)this).InitializeWeightsAndBiases();

        Console.WriteLine($"Создан слой с {inputSize} входами и {size} нейронами. Активируем с функцией: {activation}");
    }

    /// <summary>
    /// Метод для выполнения прямого прохода через полносвязный слой.
    /// Здесь вычисляется результат работы с весами и смещениями.
    /// </summary>
    /// <param name="input">Входной тензор для слоя.</param>
    /// <returns>Выходной тензор после прохождения через слой.</returns>
    public override Tensor Forward(Tensor input)
    {
        // Применение нормализации (если необходимо)
        input = input.Normalize();  // Поясните, нужно ли это на уровне слоя

        Console.WriteLine("Прямой проход через слой (Forward):");

        // Логирование входных данных
        Console.WriteLine("Входные данные: " + string.Join(", ", input.ToString()));

        // Вычисление выхода слоя
        Tensor output = new Tensor(new int[] { input.Shape[0], Size });

        for (int i = 0; i < Size; i++)
        {
            float sum = 0.0f;

            // Логика вычисления линейного выхода: сумма входов на веса
            for (int j = 0; j < InputSize; j++)
            {
                sum += input[j, 0] * weights[j, i];  // Используем веса для вычисления
            }

            // Логирование каждого шага вычислений
            Console.WriteLine($"Для нейрона {i}: сумма входных значений и весов = {sum}");

            // Логирование смещений
            Console.WriteLine($"Смещение для нейрона {i}: {biases[0, i]}");

            // Суммируем и добавляем смещение
            output[0, i] = sum + biases[0, i];
        }

        // Логирование выхода до активации
        Console.WriteLine("Выход до активации: " + string.Join(", ", output.ToString()));

        // Применение функции активации
        Tensor activatedOutput = ApplyActivation(output);

        // Логирование выхода после активации
        Console.WriteLine("Выход после активации: " + string.Join(", ", activatedOutput.ToString()));

        return activatedOutput;
    }

    /// <summary>
    /// Применение функции активации к тензору.
    /// </summary>
    /// <param name="input">Входной тензор.</param>
    /// <returns>Тензор с примененной функцией активации.</returns>
    private Tensor ApplyActivation(Tensor input)
    {
        Console.WriteLine("Применяем функцию активации...");

        // Пример: Применение функции активации Sigmoid
        if (Activation == "Sigmoid")
        {
            return input.ApplySigmoid();
        }

        // Если нужно добавить другие функции активации, их можно реализовать здесь.
        throw new InvalidOperationException("Функция активации не поддерживается.");
    }
}
