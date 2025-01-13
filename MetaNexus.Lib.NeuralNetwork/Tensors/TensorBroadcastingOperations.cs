using MetaNexus.Lib.NeuralNetwork.Tensors.Abstractions;

namespace MetaNexus.Lib.NeuralNetwork.Tensors
{
    public partial struct Tensor : ITensorBroadcastingOperations
    {
        public bool CanBroadcast(Tensor other)
        {
            int maxLength = Math.Max(_shape.Length, other._shape.Length);

            for (int i = 0; i < maxLength; i++)
            {
                int dim1 = i < _shape.Length ? _shape[i] : 1;
                int dim2 = i < other._shape.Length ? other._shape[i] : 1;

                if (dim1 != dim2 && dim1 != 1 && dim2 != 1)
                {
                    return false;
                }
            }
            return true;
        }

        public Tensor BroadcastAdd(Tensor other)
        {
            if (!CanBroadcast(other))
            {
                throw new InvalidOperationException("Невозможно выполнить операцию с трансляцией для этих тензоров.");
            }

            Tensor expandedTensor1 = ((ITensorBroadcastingOperations)this).ExpandToBroadcast(_shape, other._shape);
            Tensor expandedTensor2 = ((ITensorBroadcastingOperations)this).ExpandToBroadcast(other._shape, _shape);

            return expandedTensor1.Add(expandedTensor2);
        }

        public Tensor BroadcastSubtract(Tensor other)
        {
            if (!CanBroadcast(other))
            {
                throw new InvalidOperationException("Невозможно выполнить операцию с трансляцией для этих тензоров.");
            }
            Tensor expandedTensor1 = ((ITensorBroadcastingOperations)this).ExpandToBroadcast(_shape, other._shape);
            Tensor expandedTensor2 = ((ITensorBroadcastingOperations)this).ExpandToBroadcast(other._shape, _shape);
            return expandedTensor1.Subtract(expandedTensor2);
        }

        public Tensor BroadcastMultiply(Tensor other)
        {
            if (!CanBroadcast(other))
            {
                throw new InvalidOperationException("Невозможно выполнить операцию с трансляцией для этих тензоров.");
            }
            Tensor expandedTensor1 = ((ITensorBroadcastingOperations)this).ExpandToBroadcast(_shape, other._shape);
            Tensor expandedTensor2 = ((ITensorBroadcastingOperations)this).ExpandToBroadcast(other._shape, _shape);
            return expandedTensor1.Multiply(expandedTensor2);
        }

        public Tensor BroadcastDivide(Tensor other)
        {
            if (!CanBroadcast(other))
            {
                throw new InvalidOperationException("Невозможно выполнить операцию с трансляцией для этих тензоров.");
            }
            Tensor expandedTensor1 = ((ITensorBroadcastingOperations)this).ExpandToBroadcast(_shape, other._shape);
            Tensor expandedTensor2 = ((ITensorBroadcastingOperations)this).ExpandToBroadcast(other._shape, _shape);
            return expandedTensor1.Divide(expandedTensor2);
        }

        Tensor ITensorBroadcastingOperations.ExpandToBroadcast(int[] sourceShape, int[] targetShape)
        {
            int totalLength = 1;
            foreach (var dim in targetShape)
            {
                totalLength *= dim;
            }
            var expandedData = new float[totalLength];

            int[] sourceIndices = new int[sourceShape.Length];
            int[] targetIndices = new int[targetShape.Length];

            for (int targetIndex = 0; targetIndex < totalLength; targetIndex++)
            {
                int tempIndex = targetIndex;
                for (int i = targetShape.Length - 1; i >= 0; i--)
                {
                    targetIndices[i] = tempIndex % targetShape[i];
                    tempIndex /= targetShape[i];
                }

                for (int i = 0; i < sourceShape.Length; i++)
                {
                    if (i >= targetShape.Length || targetIndices[i] >= sourceShape[i])
                    {
                        sourceIndices[i] = 0;
                    }
                    else
                    {
                        sourceIndices[i] = targetIndices[i];
                    }
                }

                int sourceIndex = 0;
                int multiplier = 1;
                for (int i = sourceShape.Length - 1; i >= 0; i--)
                {
                    sourceIndex += sourceIndices[i] * multiplier;
                    multiplier *= sourceShape[i];
                }

                expandedData[targetIndex] = _data[sourceIndex];
            }

            return new Tensor(targetShape, expandedData);
        }
    }
}
