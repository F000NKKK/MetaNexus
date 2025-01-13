namespace MetaNexus.Lib.NeuralNetwork.Tensor
{
    using MetaNexus.Lib.NeuralNetwork.Tensor.Abstractions;
    using System;
    using System.Numerics;

    /// <summary>
    /// Фабрика для создания тензоров, использующих CPU или GPU (CUDA/OpenCL).
    /// </summary>
    public class TensorDeviceFactory<T> : ITensorDeviceFactory<T> where T : INumber<T>
    {
        /// <summary>
        /// Создает тензор, который будет использовать GPU или CPU для вычислений.
        /// В зависимости от указанной платформы, тензор будет использовать соответствующие технологии.
        /// </summary>
        /// <param name="platform">Платформа, на которой будет работать тензор. Пример значений: "CUDA", "OpenCL", "CPU".</param>
        /// <param name="shape">Форма тензора.</param>
        /// <returns>Интерфейс тензора для работы с выбранной платформой.</returns>
        public ITensorDevice<T> CreateTensorGPU(string platform, int[] shape)
        {
            if (shape == null || shape.Length == 0)
            {
                throw new ArgumentException("Размерность тензора должна быть больше нуля.", nameof(shape));
            }

            switch (platform.ToUpper())
            {
                case "CUDA":
                    return new TensorCUDA<T>(shape);
                case "OPENCL":
                    return new TensorOpenCL<T>(shape);
                case "CPU":
                    return new TensorCPU<T>(shape); 
                default:
                    throw new NotSupportedException($"Платформа {platform} не поддерживается.");
            }
        }
    }
}
