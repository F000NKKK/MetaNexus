namespace MetaNexus.Lib.Metrics.Models
{
    /// <summary>
    /// Событие метрики.
    /// </summary>
    public class RawMetric
    {
        private string _name;
        private IEnumerable<RawMetricLabel> _labels;
        private double _value;

        /// <summary>
        /// Название метрики.
        /// </summary>
        public string Name
        {
            get => _name;
            protected set
            {
                if (string.IsNullOrWhiteSpace(value))
                    throw new ArgumentNullException(nameof(value), "Name cannot be null or empty.");
                _name = value;
            }
        }

        /// <summary>
        /// Метки метрики (ключ-значение).
        /// </summary>
        public IEnumerable<RawMetricLabel>? Labels
        {
            get => _labels;
            protected set
            {
                if (value == null)
                    _labels = Array.Empty<RawMetricLabel>();
                else
                    _labels = value.ToArray();
            }
        }

        /// <summary>
        /// Значение метрики.
        /// </summary>
        public double Value
        {
            get => _value;
            protected set
            {
                if (double.IsNaN(value) || double.IsInfinity(value))
                    throw new ArgumentException("Value must be a valid number.", nameof(value));
                _value = value;
            }
        }

#pragma warning disable CS8618
        /// <summary>
        /// Конструктор для создания события метрики.
        /// </summary>
        /// <param name="name">Название метрики.</param>
        /// <param name="labels">Метки (ключ-значение).</param>
        /// <param name="value">Значение метрики.</param>
        public RawMetric(string name, IEnumerable<RawMetricLabel>? labels, double value)
        {
            Name = name;
            Labels = labels;
            Value = value;
        }

        /// <summary>
        /// Конструктор для десериализации.
        /// </summary>
        protected RawMetric() { }
#pragma warning restore CS8618
    }
}