using System.Numerics;

namespace MetaNexus.Lib.NeuralNetwork.Tensor.Abstractions
{
    /// <summary>
    /// Интерфейс для работы с тензорами на GPU.
    /// Предоставляет методы для передачи данных на GPU, выполнения вычислений и извлечения данных с устройства.
    /// </summary>
    public interface ITensorDevice<T> where T : INumber<T>
    {
        /// <summary>
        /// Переносит данные на Device.
        /// Используется для загрузки данных из памяти хоста в память устройства.
        /// </summary>
        /// <param name="data">Массив данных, который нужно перенести на GPU.</param>
        void TransferToDevice(T[] data);

        /// <summary>
        /// Выполняет ядро на Device.
        /// Выполняет вычисления на Device с использованием указанного кода ядра.
        /// </summary>
        /// <param name="kernelCode">Исходный код ядра для выполнения на Device. Ядро может быть написано на CUDA, C или OpenCL.</param>
        void ExecuteKernel(string kernelCode);

        /// <summary>
        /// Извлекает данные с Device.
        /// Возвращает данные, вычисленные на Device, в память хоста.
        /// </summary>
        /// <returns>Массив данных, полученных с Device.</returns>
        T[] GetDataFromDevice();
    }
}
