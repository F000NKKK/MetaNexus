# MetaNexus

**MetaNexus** - это библиотека для создания и работы с нейронными сетями, написанная на C#. Она предоставляет гибкие механизмы для настройки слоев, обучения и работы с нейросетями, а также включает поддержку загрузки и сохранения весов в бинарном формате.

## Описание

MetaNexus позволяет пользователю создавать нейронные сети с конфигурацией, задаваемой через JSON, а также загружать веса и смещения из бинарных файлов. Это позволяет эффективно работать с уже обученными сетями, а также с настраиваемыми сетями для различных задач машинного обучения.

## Возможности

- **Конфигурация нейронной сети**: Создание нейронной сети через JSON-конфигурацию, что позволяет задавать различные типы слоев и их параметры.
- **Загрузка весов**: Нейронная сеть может загружать веса и смещения из бинарных файлов.
- **Прогнозирование**: Нейронная сеть может быть использована для выполнения прогноза с использованием метода прямого прохода.
- **Сохранение весов**: Все веса и смещения могут быть сохранены в бинарном формате для дальнейшего использования.

## Установка

Для использования этой библиотеки достаточно добавить её в проект C#:

1. Склонируйте репозиторий:
   ```bash
   git clone https://github.com/F000NKKK/MetaNexus.git
   ```
2. Добавьте проект в решение в Visual Studio или другом IDE, поддерживающем .NET.
3. Скомпилируйте проект, чтобы получить библиотеку.

## Пример использования

```C#
using MetaNexus.Lib.NeuralNetwork.ML;
using System;

namespace Example
{
    class Program
    {
        static void Main(string[] args)
        {
            // Пример создания нейронной сети из JSON конфигурации
            string jsonConfig = @"{
                'Layers': [
                    { 'Type': 'input', 'Size': 3 },
                    { 'Type': 'dense', 'Size': 5, 'Activation': 'relu' },
                    { 'Type': 'dense', 'Size': 2, 'Activation': 'sigmoid' }
                ]
            }";
            
            var neuralNetwork = new NeuralNetwork(jsonConfig);
            
            // Прогнозирование
            float[] input = { 0.5f, 0.1f, 0.7f };
            float[] output = neuralNetwork.Predict(input);
            
            Console.WriteLine("Output: ");
            foreach (var value in output)
            {
                Console.WriteLine(value);
            }

            // Сохранение весов в бинарный файл
            neuralNetwork.SaveWeights("network_weights.bin");
            
            // Загрузка весов из бинарного файла
            var neuralNetworkWithWeights = new NeuralNetwork(jsonConfig, "network_weights.bin", loadWeights: true);
        }
    }
}
```

## Зависимости

Newtonsoft.Json - для работы с JSON конфигурациями. Убедитесь, что пакет установлен в вашем проекте:
```bash
Install-Package Newtonsoft.Json
```

# Структура проекта

Проект состоит из нескольких основных частей:

* MetaNexus.Lib.NeuralNetwork.ML: Основная библиотека для работы с нейросетями, включает классы для слоев, нейронных сетей и их конфигурации.
* MetaNexus.Lib.NeuralNetwork.ML.Layers.Abstractions: Абстракции для слоев нейросети.
* MetaNexus.Lib.NeuralNetwork.ML.Models: Модели, которые описывают конфигурацию сети.

## Лицензия

Этот проект распространяется под лицензией **AGPL-3.0**. Подробнее см. в файле [LICENSE](https://github.com/F000NKKK/MetaNexus/blob/main/LICENSE.txt).
