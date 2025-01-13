using System.Numerics;

namespace MetaNexus.Lib.NeuralNetwork.Tensor.Abstractions
{
    /// <summary>
    /// Интерфейс для создания тензоров, использующих GPU.
    /// Предоставляет метод для создания тензоров, которые могут работать на различных платформах GPU, таких как CUDA и OpenCL.
    /// </summary>
    public interface ITensorGPUFactory<T> where T : INumber<T>
    {
        /// <summary>
        /// Создает тензор, который будет использовать GPU для вычислений.
        /// В зависимости от указанной платформы, тензор будет использовать соответствующие технологии (например, CUDA или OpenCL).
        /// </summary>
        /// <param name="platform">Платформа, на которой будет работать тензор. Пример значений: "CUDA" или "OpenCL".</param>
        /// <param name="size">Размер тензора.</param>
        /// <returns>Интерфейс тензора, который предоставляет методы для работы с GPU.</returns>
        ITensorGPU<T> CreateTensorGPU(string platform, int size);
    }
}
