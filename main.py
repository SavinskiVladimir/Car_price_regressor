import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import names

#функция вывода большого списка на экран
def out(a):
      count = 0
      for i in sorted(a):
            if count != 7:
                  print(i, end=', ')
                  count += 1
            else:
                  print()
                  count = 0
                  print(i, end=', ')
      print()

# получаем данные для обучения регрессора из файла car_data.csv
data = pd.read_csv("car_data.csv")

# обучение алгоритма

# приводим строковые данные к численному формату (обучающие)
brand_encoder = LabelEncoder()
model_encoder = LabelEncoder()
city_encoder = LabelEncoder()
fuel_encoder = LabelEncoder()
transmission_encoder = LabelEncoder()
drive_encoder = LabelEncoder()

data['car_brand'] = brand_encoder.fit_transform(data['car_brand'])
data['car_model'] = model_encoder.fit_transform(data['car_model'])
data['car_city'] = city_encoder.fit_transform(data['car_city'])
data['car_fuel'] = fuel_encoder.fit_transform(data['car_fuel'])
data['car_transmission'] = transmission_encoder.fit_transform(data['car_transmission'])
data['car_drive'] = drive_encoder.fit_transform(data['car_drive'])

# разделяем данные на факторы и отклики
factors = data[['car_brand', 'car_model', 'car_city', 'car_fuel', 'car_transmission', 'car_drive', 'car_mileage',
            'car_engine_capacity', 'car_engine_hp', 'car_age']]
response = data['car_price']

# производим нормализацию данных (обучающих)
scaler = MinMaxScaler()
factors = scaler.fit_transform(factors)

# разделяем на обучающую и тестовую выборки
f_train, f_test, r_train, r_test = train_test_split(factors, response, test_size=0.19999998, random_state=0)

# регрессия случайным лесом
regressor = RandomForestRegressor()
regressor.fit(f_train, r_train)

# запуск системы
print("Интеллектуальная система определения \n"
            "стоимости автомобиля готова к работе\n"
            "Введите команду:\n"
            "help - информация по порядку и принципу ввода данных\n"
            "enter - переход к вводу данных\n"
            "exit - завершение работы системы")
while True:
      flag_start = True
      while flag_start:
            print("Команда: ", end='')
            command = input()
            if command != "help" and command != "enter" and command != "exit":
                  print("Ввод некорректен, повторите его")
            else:
                  flag_start = False
      if command == "help":
            print("Для оценки стоимости автомобиля используются следующие параметры:\n"
                  "1) Производитель: предлагается выбрать один вариант из предложенного списка, например, Volkswagen\n"
                  "2) Модель: предлагается выбрать один вариант из предложенного списка, например, Polo\n"
                  "3) Город продажи: предлагается выбрать один вариант из предложенного списка, например, Москва\n"
                  "4) Вид топлива: предлагается выбрать один вариант из предложенного списка, например, бензин\n"
                  "5) Вид трансмиссии: предлагается выбрать один вариант из предложенного списка, например, автоматическая\n"
                  "6) Вид привода: предлагается выбрать один вариант из предложенного списка, например, передний\n"
                  "7) Пробег: необходимо ввести целое число километров, например, 75000\n"
                  "8) Рабочий объём двигателя: необходимо ввести дробное число литров, разделяя разряды точкой, например, 1.5\n"
                  "9) Мощность двигателя: необходимо ввести целое число лошадиных сил, например, 110\n"
                  "10) Год производства автомобиля: необоходимо ввести целоче число, например, 2018\n")
      elif command == "enter":
            flag_glob = True
            while flag_glob:
                  # сбор данных пользователя
                  flag = False
                  out(names.brands)
                  while (flag != True):
                        print('выберите марку автомобиля: ', end='')
                        cb = input()
                        if cb in names.brands:
                              flag = True
                        else:
                              print('данные введены некорректно, \nповторите ввод')
                  print()

                  flag = False
                  if isinstance(names.models_by_brand[cb], str):
                        print(names.models_by_brand[cb])
                  else:
                        out(names.models_by_brand[cb])
                  while (flag != True):
                        print('выберите модель автомобиля: ', end='')
                        cm = input()
                        if cm in names.models_by_brand[cb]:
                              flag = True
                        else:
                              print('данные введены некорректно, \nповторите ввод')
                  print()

                  flag = False
                  out(names.cities.keys())
                  while (flag != True):
                        print('выберите город продажи: ', end='')
                        s = input()
                        if s in names.cities.keys():
                              cci = names.cities[s]
                              flag = True
                        else:
                              print('данные введены некорректно, \nповторите ввод')
                  print()

                  flag = False
                  out(names.engine_type.keys())
                  while (flag != True):
                        print('выберите вид топлива: ', end='')
                        s = input()
                        if s in names.engine_type.keys():
                              cf = names.engine_type[s]
                              flag = True
                        else:
                              print('данные введены некорректно, \nповторите ввод')
                  print()

                  flag = False
                  out(names.transmission_type.keys())
                  while (flag != True):
                        print('выберите вид кпп: ', end='')
                        s = input()
                        if s in names.transmission_type.keys():
                              ct = names.transmission_type[s]
                              flag = True
                        else:
                              print('данные введены некорректно, \nповторите ввод')
                  print()

                  flag = False
                  out(names.drive_type.keys())
                  while (flag != True):
                        print('выберите вид привода: ', end='')
                        s = input()
                        if s in names.drive_type.keys():
                              cd = names.drive_type[s]
                              flag = True
                        else:
                              print('данные введены некорректно, \nповторите ввод')
                  print()

                  flag = False
                  while (flag != True):
                        print('введите пробег: ', end='')
                        cmi = int(input())
                        if (cmi >= 0):
                              flag = True
                        else:
                              print('данные введены некорректно, \nповторите ввод')
                  print()

                  flag = False
                  while (flag != True):
                        print('введите объем двигателя: ', end='')
                        cec = float(input())
                        if cec > 0 and cec < 8:
                              flag = True
                        else:
                              print('данные введены некорректно, \nповторите ввод')
                  print()

                  flag = False
                  while (flag != True):
                        print('введите количество лошадиных сил: ', end='')
                        chp = int(input())
                        if chp > 0 and chp < 2100:
                              flag = True
                        else:
                              print('данные введены некорректно, \nповторите ввод')
                  print()

                  flag = False
                  while (flag != True):
                        print('введите год производства автомобиля: ', end='')
                        ca = 2024 - int(input())
                        if ca >= 0 and ca < 140:
                              flag = True
                        else:
                              print('данные введены некорректно, \nповторите ввод')
                  print()

                  # словарь для данных пользователя
                  order_dict = {'car_brand': cb, 'car_model': cm, 'car_city': cci, 'car_fuel': cf,
                                'car_transmission': ct, 'car_drive': cd, 'car_mileage': cmi,
                                'car_engine_capacity': cec, 'car_engine_hp': chp, 'car_age': ca}

                  # проверка введённых данных
                  print('Ваши данные:')
                  for i in order_dict.values():
                        print(i, sep=', ', end=' ')
                  print()
                  print('Данные коректны: да/нет?')
                  key = input()
                  if (key == 'да'):
                        flag_glob = False

            # переводим данные в dataframe
            order_data = pd.DataFrame(order_dict, index=[0])

            # приводим строковые данные к численному формату (пользовательские)
            order_data['car_brand'] = brand_encoder.transform(order_data['car_brand'])
            order_data['car_model'] = model_encoder.transform(order_data['car_model'])
            order_data['car_city'] = city_encoder.transform(order_data['car_city'])
            order_data['car_fuel'] = fuel_encoder.transform(order_data['car_fuel'])
            order_data['car_transmission'] = transmission_encoder.transform(order_data['car_transmission'])
            order_data['car_drive'] = drive_encoder.transform(order_data['car_drive'])

            # проводим нормализацию данных (пользовательских)
            order_data = scaler.transform(order_data)

            # делаем прогноз
            order_prediction = regressor.predict(order_data)
            print('Оценочная стоимость автомобиля с введёнными параметрами: ', end='')
            print(int(round(*order_prediction * 0.78, -2)), '-', int(round(*order_prediction * 0.95, -2)), 'руб.')
      else:
            print("Завершение работы")
            break