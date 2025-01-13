namespace MetaNexus.Lib.NeuralNetwork.Tensor.Abstractions
{
    /// <summary>
    /// Интерфейс для работы с тензорами на GPU.
    /// Предоставляет методы для передачи данных на GPU, выполнения вычислений и извлечения данных с устройства.
    /// </summary>
    public interface ITensorGPU<T> where T : struct
    {
        /// <summary>
        /// Переносит данные на GPU.
        /// Используется для загрузки данных из памяти хоста в память устройства.
        /// </summary>
        /// <param name="data">Массив данных, который нужно перенести на GPU.</param>
        void TransferToGPU(T[] data);

        /// <summary>
        /// Выполняет ядро на GPU.
        /// Выполняет вычисления на GPU с использованием указанного кода ядра.
        /// </summary>
        /// <param name="kernelCode">Исходный код ядра для выполнения на GPU. Ядро может быть написано на CUDA C или OpenCL.</param>
        void ExecuteKernel(string kernelCode);

        /// <summary>
        /// Извлекает данные с GPU.
        /// Возвращает данные, вычисленные на GPU, в память хоста.
        /// </summary>
        /// <returns>Массив данных, полученных с GPU.</returns>
        T[] GetDataFromGPU();
    }
}
