namespace MetaNexus.Lib.Metrics.Models
{
    /// <summary>
    /// Событие метрики.
    /// </summary>
    public class Metric
    {
        private string _name;
        private IEnumerable<MetricLabel> _labels;
        private double _value;
        private MetricType _type;

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
        public IEnumerable<MetricLabel>? Labels
        {
            get => _labels;
            protected set
            {
                if (value == null)
                    _labels = Array.Empty<MetricLabel>();
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

        /// <summary>
        /// Тип метрики.
        /// </summary>
        public MetricType Type
        {
            get => _type;
            protected set => _type = value;
        }

#pragma warning disable CS8618
        /// <summary>
        /// Конструктор для создания события метрики.
        /// </summary>
        /// <param name="name">Название метрики.</param>
        /// <param name="type">Тип метрики.</param>
        /// <param name="labels">Метки (ключ-значение).</param>
        /// <param name="value">Значение метрики.</param>
        public Metric(string name, MetricType type, IEnumerable<MetricLabel>? labels, double value)
        {
            Name = name;
            Type = type;
            Labels = labels;
            Value = value;
        }

        /// <summary>
        /// Конструктор для десериализации.
        /// </summary>
        protected Metric() { }
#pragma warning restore CS8618
    }
}