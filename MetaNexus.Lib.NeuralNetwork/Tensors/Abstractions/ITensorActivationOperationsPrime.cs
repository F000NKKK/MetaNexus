namespace MetaNexus.Lib.NeuralNetwork.Tensors.Abstractions
{
    /// <summary>
    /// Интерфейс для применения производных функций активации к тензорам.
    /// </summary>
    internal interface ITensorActivationOperationsPrime
    {
        /// <summary>
        /// Применение производной от функции активации Sigmoid к тензору.
        /// Возвращает тензор, где каждый элемент представляет собой производную функции Sigmoid.
        /// </summary>
        /// <returns>Результат применения производной от Sigmoid.</returns>
        Tensor ApplySigmoidPrime();

        /// <summary>
        /// Применение производной от функции активации Tanh (гиперболический тангенс) к тензору.
        /// Возвращает тензор, где каждый элемент представляет собой производную функции Tanh.
        /// </summary>
        /// <returns>Результат применения производной от Tanh.</returns>
        Tensor ApplyTanhPrime();

        /// <summary>
        /// Применение производной от функции активации ReLU (Rectified Linear Unit) к тензору.
        /// Возвращает тензор, где каждый элемент представляет собой производную функции ReLU.
        /// </summary>
        /// <returns>Результат применения производной от ReLU.</returns>
        Tensor ApplyReLUPrime();

        /// <summary>
        /// Применение производной от функции активации Leaky ReLU к тензору.
        /// Возвращает тензор, где каждый элемент представляет собой производную функции Leaky ReLU.
        /// </summary>
        /// <param name="alpha">Параметр для регулировки наклона для отрицательных значений.</param>
        /// <returns>Результат применения производной от Leaky ReLU.</returns>
        Tensor ApplyLeakyReLUPrime(float alpha);

        /// <summary>
        /// Применение производной от функции активации Softplus к тензору.
        /// Возвращает тензор, где каждый элемент представляет собой производную функции Softplus.
        /// </summary>
        /// <returns>Результат применения производной от Softplus.</returns>
        Tensor ApplySoftplusPrime();

        /// <summary>
        /// Применение производной от функции активации Swish к тензору.
        /// Возвращает тензор, где каждый элемент представляет собой производную функции Swish.
        /// </summary>
        /// <returns>Результат применения производной от Swish.</returns>
        Tensor ApplySwishPrime();

        /// <summary>
        /// Применение производной от функции активации GELU (Gaussian Error Linear Unit) к тензору.
        /// Возвращает тензор, где каждый элемент представляет собой производную функции GELU.
        /// </summary>
        /// <returns>Результат применения производной от GELU.</returns>
        Tensor ApplyGELUPrime();

        /// <summary>
        /// Применение производной от функции активации Hard Sigmoid к тензору.
        /// Возвращает тензор, где каждый элемент представляет собой производную функции Hard Sigmoid.
        /// </summary>
        /// <returns>Результат применения производной от Hard Sigmoid.</returns>
        Tensor ApplyHardSigmoidPrime();

        /// <summary>
        /// Применение производной от функции активации Hard Tanh к тензору.
        /// Возвращает тензор, где каждый элемент представляет собой производную функции Hard Tanh.
        /// </summary>
        /// <returns>Результат применения производной от Hard Tanh.</returns>
        Tensor ApplyHardTanhPrime();

        /// <summary>
        /// Применение производной от функции активации Mish к тензору.
        /// Возвращает тензор, где каждый элемент представляет собой производную функции Mish.
        /// </summary>
        /// <returns>Результат применения производной от Mish.</returns>
        Tensor ApplyMishPrime();
    }
}
