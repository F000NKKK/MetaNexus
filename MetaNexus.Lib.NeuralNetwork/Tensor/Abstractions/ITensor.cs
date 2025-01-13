namespace MetaNexus.Lib.NeuralNetwork.Tensor.Abstractions
{
    /// <summary>
    /// Интерфейс для работы с тензорами, поддерживающий доступ к данным,
    /// а также выполнение различных операций над тензорами.
    /// </summary>
    internal interface ITensor : ITensorOperations
    {
        /// <summary>
        /// Индексатор для доступа к элементам тензора по заданным индексам.
        /// </summary>
        /// <param name="indices">Массив индексов для доступа к элементу тензора.</param>
        /// <returns>Значение элемента тензора по указанным индексам.</returns>
        float this[params int[] indices] { get; set; }

        /// <summary>
        /// Получение формы тензора (размерности для каждого измерения).
        /// </summary>
        /// <returns>Массив целых чисел, представляющих форму тензора.</returns>
        int[] Shape { get; }

        /// <summary>
        /// Получение ранга тензора (количество измерений).
        /// </summary>
        /// <returns>Ранг тензора.</returns>
        int Rank { get; }

        /// <summary>
        /// Получение общего числа элементов в тензоре (произведение размеров всех измерений).
        /// </summary>
        /// <returns>Общее количество элементов в тензоре.</returns>
        int Size { get; }

        /// <summary>
        /// Проверка, является ли тензор пустым (не содержит элементов).
        /// </summary>
        /// <returns>True, если тензор пустой, иначе - False.</returns>
        bool IsEmpty();

        /// <summary>
        /// Получение копии тензора.
        /// </summary>
        /// <returns>Новая копия тензора с теми же данными.</returns>
        Tensor Clone();

        /// <summary>
        /// Получение плоского представления тензора как одномерного массива.
        /// </summary>
        /// <returns>Массив значений всех элементов тензора.</returns>
        float[] Flatten();

        /// <summary>
        /// Применение функции ко всем элементам тензора.
        /// </summary>
        /// <param name="func">Функция для применения к каждому элементу тензора.</param>
        /// <returns>Новый тензор с результатами применения функции.</returns>
        Tensor Apply(Func<float, float> func);
    }
}
