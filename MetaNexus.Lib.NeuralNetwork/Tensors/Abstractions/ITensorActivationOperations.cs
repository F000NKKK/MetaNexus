namespace MetaNexus.Lib.NeuralNetwork.Tensors.Abstractions
{
    /// <summary>
    /// Интерфейс для применения функций активации к тензорам.
    /// </summary>
    internal interface ITensorActivationOperations
    {
        /// <summary>
        /// Применение функции активации ReLU (Rectified Linear Unit) к тензору.
        /// Возвращает тензор, где все значения меньше нуля заменяются на ноль.
        /// </summary>
        /// <returns>Результат применения ReLU.</returns>
        Tensor ApplyReLU();

        /// <summary>
        /// Применение функции активации Leaky ReLU к тензору.
        /// Возвращает тензор, где значения меньше нуля заменяются на alpha * x.
        /// </summary>
        /// <param name="alpha">Параметр для регулировки наклона для отрицательных значений.</param>
        /// <returns>Результат применения Leaky ReLU.</returns>
        Tensor ApplyLeakyReLU(float alpha);

        /// <summary>
        /// Применение функции активации сигмоид (Sigmoid) к тензору.
        /// Возвращает тензор, где каждый элемент преобразуется с помощью функции Sigmoid.
        /// </summary>
        /// <returns>Результат применения сигмоиды.</returns>
        Tensor ApplySigmoid();

        /// <summary>
        /// Применение функции активации tanh (гиперболический тангенс) к тензору.
        /// Возвращает тензор, где каждый элемент преобразуется с помощью функции tanh.
        /// </summary>
        /// <returns>Результат применения tanh.</returns>
        Tensor ApplyTanh();

        /// <summary>
        /// Применение функции активации Softmax к тензору.
        /// Преобразует элементы тензора в вероятности, сумма которых равна 1.
        /// Обычно используется на выходном слое в задачах классификации.
        /// </summary>
        /// <returns>Результат применения Softmax.</returns>
        Tensor ApplySoftmax();

        /// <summary>
        /// Применение функции активации Swish к тензору.
        /// Возвращает тензор, где каждый элемент преобразуется с помощью функции Swish.
        /// </summary>
        /// <returns>Результат применения Swish.</returns>
        Tensor ApplySwish();

        /// <summary>
        /// Применение функции активации ELU (Exponential Linear Unit) к тензору.
        /// Возвращает тензор, где все значения меньше нуля заменяются на экспоненциальную функцию.
        /// </summary>
        /// <param name="alpha">Параметр для регулировки наклона для отрицательных значений.</param>
        /// <returns>Результат применения ELU.</returns>
        Tensor ApplyELU(float alpha);

        /// <summary>
        /// Применение функции активации Softplus к тензору.
        /// Возвращает тензор, где каждый элемент преобразуется с помощью функции Softplus.
        /// </summary>
        /// <returns>Результат применения Softplus.</returns>
        Tensor ApplySoftplus();

        /// <summary>
        /// Применение функции активации Hard Sigmoid к тензору.
        /// Возвращает тензор, где значения ограничены в диапазоне от 0 до 1.
        /// </summary>
        /// <returns>Результат применения Hard Sigmoid.</returns>
        Tensor ApplyHardSigmoid();

        /// <summary>
        /// Применение функции активации GELU (Gaussian Error Linear Unit) к тензору.
        /// Возвращает тензор, где каждый элемент преобразуется с помощью функции GELU.
        /// </summary>
        /// <returns>Результат применения GELU.</returns>
        Tensor ApplyGELU();

        /// <summary>
        /// Применение функции активации Hard Tanh к тензору.
        /// Возвращает тензор, где все значения ограничены в диапазоне от -1 до 1.
        /// </summary>
        /// <returns>Результат применения Hard Tanh.</returns>
        Tensor ApplyHardTanh();

        /// <summary>
        /// Применение функции активации Mish к тензору.
        /// Возвращает тензор, где каждый элемент преобразуется с помощью функции Mish.
        /// </summary>
        /// <returns>Результат применения Mish.</returns>
        Tensor ApplyMish();

        /// <summary>
        /// Применение тождественной функции (Identity) к тензору.
        /// Возвращает сам тензор без изменений.
        /// </summary>
        /// <returns>Сам тензор без изменений.</returns>
        Tensor ApplyIdentity();

        /// <summary>
        /// Применение функции активации SoftSign к тензору.
        /// Возвращает тензор, где каждый элемент преобразуется с помощью функции SoftSign.
        /// </summary>
        /// <returns>Результат применения SoftSign.</returns>
        Tensor ApplySoftSign();
    }
}
