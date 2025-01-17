﻿namespace MetaNexus.Lib.NeuralNetwork.Tensors.Abstractions
{
    /// <summary>
    /// Интерфейс, который определяет операции трансляции (broadcasting) для тензоров.
    /// </summary>
    internal interface ITensorBroadcastingOperations
    {
        /// <summary>
        /// Проверяет, может ли текущий тензор выполнять операцию трансляции с другим тензором.
        /// Тензоры считаются совместимыми для трансляции, если их размерности соответствуют друг другу по определенным правилам.
        /// </summary>
        /// <param name="other">Другой тензор для проверки совместимости трансляции.</param>
        /// <returns>Возвращает <c>true</c>, если тензоры могут быть трансляции, иначе <c>false</c>.</returns>
        bool CanBroadcast(Tensor other);

        /// <summary>
        /// Выполняет операцию сложения с трансляцией с другим тензором.
        /// Тензоры с различными размерностями будут расширены (с трансляцией) до совместимой формы для выполнения операции.
        /// </summary>
        /// <param name="other">Другой тензор для сложения.</param>
        /// <returns>Результат сложения с трансляцией.</returns>
        Tensor BroadcastAdd(Tensor other);

        /// <summary>
        /// Выполняет операцию вычитания с трансляцией с другим тензором.
        /// Тензоры с различными размерностями будут расширены (с трансляцией) до совместимой формы для выполнения операции.
        /// </summary>
        /// <param name="other">Другой тензор для вычитания.</param>
        /// <returns>Результат вычитания с трансляцией.</returns>
        Tensor BroadcastSubtract(Tensor other);

        /// <summary>
        /// Выполняет операцию умножения с трансляцией с другим тензором.
        /// Тензоры с различными размерностями будут расширены (с трансляцией) до совместимой формы для выполнения операции.
        /// </summary>
        /// <param name="other">Другой тензор для умножения.</param>
        /// <returns>Результат умножения с трансляцией.</returns>
        Tensor BroadcastMultiply(Tensor other);

        /// <summary>
        /// Выполняет операцию деления с трансляцией с другим тензором.
        /// Тензоры с различными размерностями будут расширены (с трансляцией) до совместимой формы для выполнения операции.
        /// </summary>
        /// <param name="other">Другой тензор для деления.</param>
        /// <returns>Результат деления с трансляцией.</returns>
        Tensor BroadcastDivide(Tensor other);

        /// <summary>
        /// Расширяет данные одного тензора до формы другого тензора с использованием трансляции.
        /// Используется для приведения меньшего тензора к размерности большего для выполнения операций.
        /// </summary>
        /// <param name="targetShape">Целевая размерность тензора, к которой нужно привести исходный тензор.</param>
        /// <returns>Новый тензор, который имеет форму целевого тензора с расширенными данными.</returns>
        Tensor ExpandToBroadcast(int[] targetShape);
    }
}
