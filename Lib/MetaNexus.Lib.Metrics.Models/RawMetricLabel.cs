namespace MetaNexus.Lib.Metrics.Models
{
    /// <summary>
    /// Метка метрики.
    /// </summary>
    public class RawMetricLabel
    {
        private string _key;
        private string _value;

        /// <summary>
        /// Значение метки.
        /// </summary>
        public string Value
        {
            get => _value;
            set
            {
                if (string.IsNullOrWhiteSpace(value))
                    throw new ArgumentNullException(nameof(value), "Value cannot be null or empty.");
                _value = value;
            }
        }

        /// <summary>
        /// Ключ метки.
        /// </summary>
        public string Key
        {
            get => _key;
            set
            {
                if (string.IsNullOrWhiteSpace(value))
                    throw new ArgumentNullException(nameof(value), "Key cannot be null or empty.");
                _key = value;
            }
        }

#pragma warning disable CS8618
        /// <summary>
        /// Конструктор для создания метки.
        /// </summary>
        /// <param name="key">Ключ метки.</param>
        /// <param name="value">Значение метки.</param>
        public RawMetricLabel(string key, string value)
        {
            Key = key;
            Value = value;
        }

        /// <summary>
        /// Конструктор для десериализации.
        /// </summary>
        protected RawMetricLabel() { }
#pragma warning restore CS8618
    }
}