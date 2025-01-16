using MetaNexus.Lib.NeuralNetwork.Tensors.Abstractions;

namespace MetaNexus.Lib.NeuralNetwork.Tensors
{
    public partial struct Tensor : ITensorArithmeticOperations
    {
        public Tensor Add(Tensor other)
        {
            return ((ITensorElementWiseOperations)this).ElementWiseOperation(other, (a, b) => a + b);
        }

        public Tensor Add(float scalar)
        {
            return ((ITensorElementWiseOperations)this).ElementWiseOperation(scalar, (a, b) => a + b);
        }

        public Tensor Divide(Tensor other)
        {
            if (other._data.Span.Contains(0f))
                throw new DivideByZeroException("Cannot divide by a tensor containing zero.");

            return ((ITensorElementWiseOperations)this).ElementWiseOperation(other, (a, b) => a / b);
        }

        public Tensor Divide(float scalar)
        {
            if (scalar == 0f)
                throw new DivideByZeroException("Cannot divide by zero scalar.");

            return ((ITensorElementWiseOperations)this).ElementWiseOperation(scalar, (a, b) => a / b);
        }

        public Tensor Multiply(Tensor other)
        {
            return ((ITensorElementWiseOperations)this).ElementWiseOperation(other, (a, b) => a * b);
        }

        public Tensor Multiply(float scalar)
        {
            return ((ITensorElementWiseOperations)this).ElementWiseOperation(scalar, (a, b) => a * b);
        }

        public Tensor Subtract(Tensor other)
        {
            return ((ITensorElementWiseOperations)this).ElementWiseOperation(other, (a, b) => a - b);
        }

        public Tensor Subtract(float scalar)
        {
            return ((ITensorElementWiseOperations)this).ElementWiseOperation(scalar, (a, b) => a - b);
        }

        // Перегрузки операторов
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
