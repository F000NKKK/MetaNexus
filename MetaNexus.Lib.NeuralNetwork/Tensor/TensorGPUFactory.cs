namespace MetaNexus.Lib.NeuralNetwork.Tensor
{
    using MetaNexus.Lib.NeuralNetwork.Tensor.Abstractions;
    using System;
    using System.Numerics;

    /// <summary>
    /// Фабрика для создания тензоров, использующих CPU или GPU (CUDA/OpenCL).
    /// </summary>
    public class TensorGPUFactory<T> : ITensorGPUFactory<T> where T : INumber<T>
    {
        /// <summary>
        /// Создает тензор, который будет использовать GPU или CPU для вычислений.
        /// В зависимости от указанной платформы, тензор будет использовать соответствующие технологии.
        /// </summary>
        /// <param name="platform">Платформа, на которой будет работать тензор. Пример значений: "CUDA", "OpenCL", "CPU".</param>
        /// <param name="size">Размер тензора.</param>
        /// <returns>Интерфейс тензора для работы с выбранной платформой.</returns>
        public ITensorGPU<T> CreateTensorGPU(string platform, int size)
        {
            if (size <= 0)
            {
                throw new ArgumentException("Размер тензора должен быть больше нуля.", nameof(size));
            }

            switch (platform.ToUpper())
            {
                case "CUDA":
                    return new TensorCUDA<T>(size);
                case "OPENCL":
                    return new TensorOpenCL<T>(size);
                case "CPU":
                    return new Tensor<T>(size); 
                default:
                    throw new NotSupportedException($"Платформа {platform} не поддерживается.");
            }
        }
    }
}
