using MetaNexus.Lib.NeuralNetwork.Tensor.Abstractions;

namespace MetaNexus.Lib.NeuralNetwork.Tensor
{
    public partial struct Tensor : ITensorArithmeticOperations
    {
        private void CheckDimensions(Tensor other)
        {
            if (_data.Length != other._data.Length)
                throw new InvalidOperationException("Размерности тензоров не совпадают.");
        }

        Tensor ITensorArithmeticOperations.Add(Tensor other)
        {
            CheckDimensions(other);
            var resultData = new float[_data.Length];
            for (int i = 0; i < _data.Length; i++)
            {
                resultData[i] = _data[i] + other._data[i];
            }
            return new Tensor(_shape, resultData);
        }

        Tensor ITensorArithmeticOperations.Add(float scalar)
        {
            var resultData = new float[_data.Length];
            for (int i = 0; i < _data.Length; i++)
            {
                resultData[i] = _data[i] + scalar;
            }
            return new Tensor(_shape, resultData);
        }

        Tensor ITensorArithmeticOperations.Divide(Tensor other)
        {
            CheckDimensions(other);
            var resultData = new float[_data.Length];
            for (int i = 0; i < _data.Length; i++)
            {
                resultData[i] = _data[i] / other._data[i];
            }
            return new Tensor(_shape, resultData);
        }

        Tensor ITensorArithmeticOperations.Divide(float scalar)
        {
            if (scalar == 0)
                throw new DivideByZeroException("Деление на ноль недопустимо.");

            var resultData = new float[_data.Length];
            for (int i = 0; i < _data.Length; i++)
            {
                resultData[i] = _data[i] / scalar;
            }
            return new Tensor(_shape, resultData);
        }

        Tensor ITensorArithmeticOperations.Multiply(Tensor other)
        {
            CheckDimensions(other);
            var resultData = new float[_data.Length];
            for (int i = 0; i < _data.Length; i++)
            {
                resultData[i] = _data[i] * other._data[i];
            }
            return new Tensor(_shape, resultData);
        }

        Tensor ITensorArithmeticOperations.Multiply(float scalar)
        {
            var resultData = new float[_data.Length];
            for (int i = 0; i < _data.Length; i++)
            {
                resultData[i] = _data[i] * scalar;
            }
            return new Tensor(_shape, resultData);
        }

        Tensor ITensorArithmeticOperations.Subtract(Tensor other)
        {
            CheckDimensions(other);
            var resultData = new float[_data.Length];
            for (int i = 0; i < _data.Length; i++)
            {
                resultData[i] = _data[i] - other._data[i];
            }
            return new Tensor(_shape, resultData);
        }

        Tensor ITensorArithmeticOperations.Subtract(float scalar)
        {
            var resultData = new float[_data.Length];
            for (int i = 0; i < _data.Length; i++)
            {
                resultData[i] = _data[i] - scalar;
            }
            return new Tensor(_shape, resultData);
        }

        public static Tensor operator +(Tensor tensor1, Tensor tensor2)
        {
            return ((ITensorArithmeticOperations)tensor1).Add(tensor2);
        }

        public static Tensor operator +(Tensor tensor, float scalar)
        {
            return ((ITensorArithmeticOperations)tensor).Add(scalar);
        }

        public static Tensor operator -(Tensor tensor1, Tensor tensor2)
        {
            return ((ITensorArithmeticOperations)tensor1).Subtract(tensor2);
        }

        public static Tensor operator -(Tensor tensor, float scalar)
        {
            return ((ITensorArithmeticOperations)tensor).Subtract(scalar);
        }

        public static Tensor operator *(Tensor tensor1, Tensor tensor2)
        {
            return ((ITensorArithmeticOperations)tensor1).Multiply(tensor2);
        }

        public static Tensor operator *(Tensor tensor, float scalar)
        {
            return ((ITensorArithmeticOperations)tensor).Multiply(scalar);
        }

        public static Tensor operator /(Tensor tensor1, Tensor tensor2)
        {
            return ((ITensorArithmeticOperations)tensor1).Divide(tensor2);
        }

        public static Tensor operator /(Tensor tensor, float scalar)
        {
            return ((ITensorArithmeticOperations)tensor).Divide(scalar);
        }
    }
}
