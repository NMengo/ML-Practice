#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from datetime import date as dte
from datetime import datetime as dt
pd.options.display.float_format = '{:,.2f}'.format
pd.options.display.max_rows = None
pd.options.display.max_columns = None
Data = pd.read_excel('Venta COTO.xlsx', sheet_name='IpadAmp')
Data = Data.iloc[:, 0:16]

# Hay columnas sin header. Se renombran todas las columnas y se eliminan las que no sirven.
Data.columns = ['Combo' , 'Oferta', 'Inicio', 'Fin', 'elim1', 'Plu', 'Descrip', 'elim2', 'elim3', 'elim4', 'elim5', 'PVP Lote', 'elim6', 'PVP Reg', 'Unidades', 'Fact']
Data = Data.drop(['elim1','elim2','elim3','elim4','elim5','elim6'], axis=1)

# Data es Ventas Combo Ipad Ampliado.
for i in range(len(Data)):
    # Primero se trae la semana del año correspondiente en notación ISO, tanto para la fecha de inicio como fin de oferta.
    # Para evitar duplicados cuando el usuario saca más de un año, se diferenció concatenó el año.
    Data.loc[i, 'S. Inicio'] = str(Data.loc[i, 'Inicio'].isocalendar()[0])+str(Data.loc[i, 'Inicio'].isocalendar()[1])
    Data.loc[i, 'S. Fin'] = str(Data.loc[i, 'Fin'].isocalendar()[0])+str(Data.loc[i, 'Fin'].isocalendar()[1])

    # Si la diferencia entre la semana de finalización y la semana de inicio es mayor a 1
    # Significa que hay una semana en el medio.
    if (Data.loc[i, 'Fin'].isocalendar()[1] - Data.loc[i, 'Inicio'].isocalendar()[1]) >= 2:
        Data.loc[i, 'S. Int'] = str(Data.loc[i, 'Inicio'].isocalendar()[0])+str(Data.loc[i, 'Inicio'].isocalendar()[1] + 1)
    else:
        Data.loc[i, 'S. Int'] = 0



    # Calcular la cantidad de días que el producto estuvo de oferta por cada semana.

    # Para el caso de Semana Inicial, basta con restarle a 8, el número de día de semana que le corresponde a esa fecha.
    # Por ejemplo: si la oferta comenzó un Domingo, de esa semana solo estuvo 1 día de oferta.
    # Domingo    --> .isocalendar()[2] = 7mo día de la semana
    Data.loc[i, 'Ds SI'] = 8 - Data.loc[i, 'Inicio'].isocalendar()[2]

    # Si la oferta comienza y termina en la misma semana, evidentemente en cantidad de días de S.Fin debe ir 0.
    # Si no, la cantidad de días es directamente el día de semana en que termina la oferta.
    # Siendo 1 = Lunes ... 7 = Domingo.
    if Data.loc[i, 'S. Inicio'] != Data.loc[i, 'S. Fin']:
        try:
            Data.loc[i, 'Ds SF'] = Data.loc[i, 'Fin'].isocalendar()[2]
        except:
            Data.loc[i, 'Ds SF'] = 0
    else:
        Data.loc[i, 'Ds SF'] = 0

    # Si existe semana intermedia, la misma siempre va a ser completa. Entonces son 7 días.
    if Data.loc[i, 'S. Int'] != 0:
        Data.loc[i, 'Ds SInt'] = 7
    else:
        Data.loc[i, 'Ds SInt'] = 0

# La facturación que trae el sistema ya tiene descontada las ofertas.
# Si se quiere saber cuál sería la facturación equivalente a las unidades vendidas:
Data['Tickeado Regular'] = Data['Unidades'] * Data['PVP Reg']

# Por lo tanto, el costo de todas las ofertas es:
Data['Costo Of'] = Data['Tickeado Regular'] - Data['Fact']

# En términos relativos, el % de descuento general.
Data['Dto'] = (Data['Costo Of'] / Data['Tickeado Regular'])*100
Data = Data[['Combo', 'Oferta', 'Inicio', 'Fin', 'Plu', 'Descrip', 'PVP Lote', 'PVP Reg', 'Unidades', 'Fact', 'S. Inicio', 'S. Int', 'S. Fin', 'Ds SI', 'Ds SInt', 'Ds SF', 'Tickeado Regular', 'Costo Of', 'Dto']
]

# En este diccionario se intenta contemplar las diferentes posibilidades de typeo cuando se carga cada oferta en sistema.
Modalidades = {'2x1':'2X1', '2 x 1':'2X1', '3x2':'3X2', '3 x 2':'3X2', '6x4':'6X4', '6 x 4':'6X4', '4x3':'4X3', '4 x 3':'4X3', '6x5':'6X5', '6 x 5':'6X5', '12x10':'12X10', '12 x 10':'12X10', '6x3':'6X3', '6 x 3':'6X3', '4x2':'4X2', '4 x 2':'4X2', '12x8':'12X8', '12 x 8':'12X8', '12x6':'12X6', '12 x 6':'12X6',
              'll6':'LL6','ll12':'LL12','ll2':'LL2','llv2':'LL2','llv12':'LL12', 'll 12':'LL12','ll12':'LL12', 'll 6':'LL6','ll6':'LL6','ll 2':'LL2','ll 4':'LL4','llv4':'LL4','ll4':'LL4',' x2':'LL2',
              ' x4':'LL4','lleva 12 paga 6':'LL12', '2x ':'LL2', 'lle 12':'LL12', 'lle 6':'LL6', 'lle 4':'LL4', 'llv6':'LL6', 'x12':'LL12'}

Descuentos = ['20','25','30','35','40','50']

Data['Modalidad'] = 0
Data['Descuento'] = 0


# Anidación de dos ciclos FOR para conseguir que por cada linea de Data, busque alguna coincidencia en Modalidades.
for i in range(len(Data)):
    for a in Modalidades:
        # Si alguna de las expresiones de Modalidades se encuentra en la descripción de la oferta,
        # Se busca del diccionario un valor estándar para representar la oferta.
        if a in Data.loc[i, 'Oferta'].lower():
            Data.loc[i, 'Modalidad'] = Modalidades.get(a)
            break
    # Si no encuentra modalidad, significa que fue un descuento directo.
    if Data.loc[i, 'Modalidad'] == 0:
        Data.loc[i, 'Modalidad'] = 'DIRECTO'

    # Si es una oferta del tipo a X b, el descuento se calcula 1 - b/a
    if 'X' in Data.loc[i, 'Modalidad']:
        Data.loc[i, 'Descuento'] = round(100-((int(Data.loc[i, 'Modalidad'].split('X')[-1]) / int(Data.loc[i, 'Modalidad'].split('X')[0])) * 100))

    # Se busca en la descripción de la oferta si ya se menciona ahi el descuento aplicado.
    else:
        for b in Descuentos:
            if b in Data.loc[i, 'Oferta'].lower():
                Data.loc[i, 'Descuento'] = b
                break
        # Si no encuentra nada y no es un a X b, se calcula el descuento con la diferencia entre Fact Real y Fact Of.
        # (mencionada más arriba).
        if Data.loc[i, 'Descuento'] == 0 and (Data.loc[i, 'Modalidad'] == 'DIRECTO' or 'LL' in Data.loc[i, 'Modalidad']):
            if Data.loc[i, 'Dto'] < 17:
                Data.loc[i, 'Descuento'] = 15
            elif Data.loc[i, 'Dto'] < 21:
                Data.loc[i, 'Descuento'] = 20
            elif Data.loc[i, 'Dto'] < 26:
                Data.loc[i, 'Descuento'] = 25
            elif Data.loc[i, 'Dto'] < 31:
                Data.loc[i, 'Descuento'] = 30
            elif Data.loc[i, 'Dto'] < 34:
                Data.loc[i, 'Descuento'] = 33
            elif Data.loc[i, 'Dto'] < 36:
                Data.loc[i, 'Descuento'] = 35
            elif Data.loc[i, 'Dto'] < 41:
                Data.loc[i, 'Descuento'] = 40
            elif Data.loc[i, 'Dto'] < 51:
                Data.loc[i, 'Descuento'] = 50
            elif Data.loc[i, 'Dto'] < 100:
                Data.loc[i, 'Descuento'] = Data.loc[i,'Dto']
        # Acá hay cierto margen de error y es por eso que, por ejemplo, se asigna 50% descuento si encuentra 51%.
        # Esto se debe a que hay pocos pero hay descuentos que acumulan. Se puede pasar de ese 50%.
        # Donde hay más margen de error es hacia abajo. Es decir, que el descuento final sea menor al esperado.
        # Por ejemplo, el 70% de la venta es con tarjeta comunidad. Es decir, que lo que se ve como "venta regular",
        # ya tiene un 15%. Básicamente si se hace un 6x5, el descuento real puede llegar a rondar el 3%.


    Data.loc[i, 'Mod-Dto'] = str(Data.loc[i, 'Modalidad']) + ' - ' + str(Data.loc[i, 'Descuento']) + '%'

Data['Descuento'] = Data['Descuento'].astype('int64')

# Data/CombosIpadAmpliado terminado.

print('25%')

# data2 tiene la venta por producto por semana.
data2 = pd.read_excel('ventas.xlsx')
data2 = data2.fillna(0)
data2.rename(columns={'Unnamed: 1':'Descrip'}, inplace=True)

# Se insertan 4 columnas en blanco por cada columna(semana) en archivo actual.
for c in range(len(data2.iloc[0, 2:])):
    data2.insert(c*5+3, 'Of?'+str(c), '',allow_duplicates=True)
    data2.insert(c*5+4, '%VAR'+str(c), '', allow_duplicates=True)
    data2.insert(c*5+5, 'Días'+str(c), '', allow_duplicates=True)
    data2.insert(c*5+6, 'Modalidad'+str(c), '', allow_duplicates=True)

# Doble ciclo FOR anidado, primero recorriendo por filas y luego por columnas.
for k, row in data2.iterrows():
    for l in range(len(data2.iloc[0, 2:])):
        try:
            # Fecha de cada semana pasada a ISO.
            fecha_col = (str(data2.columns[l*5+2].isocalendar()[0])+str(data2.columns[l*5+2].isocalendar()[1]))
            indice_fecha = data2.columns[l*5+2]
            indice_fecha_ant = data2.columns[(l-1)*5+2]

            # Se trae la información de Data.
            data2.loc[k, 'Of?'+str(l)] = 'SI' if (len(Data[(Data['Plu']==row['Producto']) & (fecha_col == Data['S. Inicio'])])) \
                                                 or (len(Data[(Data['Plu']==row['Producto']) & (fecha_col == Data['S. Int'])])) \
                                                 or (len(Data[(Data['Plu']==row['Producto']) & (fecha_col == Data['S. Fin'])])) != 0 else 'NO'

            data2.loc[k, 'Días'+str(l)] = int(Data[(Data['Plu']==row['Producto']) & (fecha_col == Data['S. Inicio'])]['Ds SI']) if (len(Data[(Data['Plu']==row['Producto']) & (fecha_col == Data['S. Inicio'])])) != 0 \
                                                 else int(Data[(Data['Plu']==row['Producto']) & (fecha_col == Data['S. Int'])]['Ds SInt']) if (len(Data[(Data['Plu']==row['Producto']) & (fecha_col == Data['S. Int'])])) != 0 \
                                                 else int(Data[(Data['Plu']==row['Producto']) & (fecha_col == Data['S. Fin'])]['Ds SF']) if (len(Data[(Data['Plu']==row['Producto']) & (fecha_col == Data['S. Fin'])])) != 0 else ''

            data2.loc[k, 'Modalidad' + str(l)] = list(Data[(Data['Plu']==row['Producto']) & (fecha_col == Data['S. Inicio'])]['Mod-Dto']) if (len(Data[(Data['Plu']==row['Producto']) & (fecha_col == Data['S. Inicio'])])) != 0 \
                                                 else list(Data[(Data['Plu'] == row['Producto']) & (fecha_col == Data['S. Int'])]['Mod-Dto']) if (len(Data[(Data['Plu'] == row['Producto']) & (fecha_col == Data['S. Int'])])) != 0 \
                                                 else list(Data[(Data['Plu'] == row['Producto']) & (fecha_col == Data['S. Fin'])]['Mod-Dto']) if (len(Data[(Data['Plu'] == row['Producto']) & (fecha_col == Data['S. Fin'])])) != 0 else ''
        except:
            continue

        try:
            data2.loc[k, '%VAR' + str(l)] = round(((data2.loc[k, indice_fecha] - data2.loc[k, indice_fecha_ant]) / data2.loc[k, indice_fecha_ant])*100)
        except:
            continue

data2 = data2.round(0)

for m in range(len(data2)): # Recorre fila por fila
    data2.loc[m, 'Cant Of'] = list(data2.loc[m, :]).count('SI') # Cuenta todos los valores "SI" de la fila.
    if (data2.loc[m, 'Cant Of'] / 25) == 0:
        data2.loc[m, 'Of/mes'] = 'Sin Ofertas'
    elif (data2.loc[m, 'Cant Of'] / 25) < 1:
        data2.loc[m, 'Of/mes'] = '1 oferta cada ' + str(round((1/(data2.loc[m, 'Cant Of'] / 25)), 0)) + ' meses'
    else:
        data2.loc[m, 'Of/mes'] = str(round((data2.loc[m, 'Cant Of'] / 25), 0)) + ' ofertas por mes'

print('50%')

# Calculándolo de esta manera, sirve solo para tener una variación promedio y esa variación aplicarla a la venta orgánica
# del momento en el que se está haciendo el análisis.
# En la mayoría de los casos, no sirve como para utilizar como parámetro de venta organica/oferta por la estacionalidad.
# Se puede evaluar hacerlo con el promedio de las variaciones ya que de esta forma hay algunos casos que pueden dar error.
# Por ejemnplo: se tiene el caso de un producto que la venta normal está ponderada hacia arriba por la venta en temporada
# Y se cruza contra la venta promedio de 25%, que solo se hicieron ofertas fuera de temporada.

# Para hacer análisis de saturación y acopio se deben buscar otros métodos.

for y in range(len(data2)): # Cada vez que recorra una fila nueva, restaura los valores a 0 porque es un producto diferente.
        Sumas = {'NoOf': 0, 'Of': 0, '17': 0, '20': 0, '25': 0, '30': 0, '33': 0, '35': 0, '40': 0, '50': 0}
        Cuentas = {'NoOf': 0, 'Of': 0, '17': 0, '20': 0, '25': 0, '30': 0, '33': 0, '35': 0, '40': 0, '50': 0}

        for z in range(len(data2.T)):
            if data2.iloc[y, z] == 'SI':
                Sumas['Of'] += data2.iloc[y, z - 1] # Suma acumulada del valor anterior al "SI", que es la venta en Oferta.
                Cuentas['Of'] += 1
            elif data2.iloc[y, z] == 'NO':
                Sumas['NoOf'] += data2.iloc[y, z - 1] # Suma acumulada del valor anterior al "NO", que es la venta regular.
                Cuentas['NoOf'] += 1
            else:
                try:
                    # Suma y cuenta para cada nivel de descuento.
                    if ('17%' or '15%') in data2.iloc[y, z]:
                        Sumas['17'] += data2.iloc[y, z - 4]
                        Cuentas['17'] += 1
                    elif '20%' in data2.iloc[y, z]:
                        Sumas['20'] += data2.iloc[y, z - 4]
                        Cuentas['20'] += 1
                    elif '25' in data2.iloc[y, z]:
                        Sumas['25'] += data2.iloc[y, z - 4]
                        Cuentas['25'] += 1
                    elif '30%' in data2.iloc[y, z]:
                        Sumas['30'] += data2.iloc[y, z - 4]
                        Cuentas['30'] += 1
                    elif '33%' in data2.iloc[y, z]:
                        Sumas['33'] += data2.iloc[y, z - 4]
                        Cuentas['33'] += 1
                    elif '35%' in data2.iloc[y, z]:
                        Sumas['35'] += data2.iloc[y, z - 4]
                        Cuentas['35'] += 1
                    elif '40%' in data2.iloc[y, z]:
                        Sumas['40'] += data2.iloc[y, z - 4]
                        Cuentas['40'] += 1
                    elif '50%' in data2.iloc[y, z]:
                        Sumas['50'] += data2.iloc[y, z - 4]
                        Cuentas['50'] += 1
                except:
                    continue

        # Antes que termine el Loop mayor, se completa data2 con lo acumulado en los diccionarios.
        # Se recorren ambos al mismo tiempo ya que son 2 variables para las mismas Keys.
        for Su, Cu in zip(Sumas,Cuentas):
            if Cuentas.get(Cu) != 0:
                data2.loc[y, 'Prom '+str(Su)] = round(Sumas.get(Su) / Cuentas.get(Cu),0)
                data2.loc[y, 'Cuenta '+str(Su)] = Cuentas.get(Cu)
                if Cuentas.get(Cu) != 'NoOf':
                    data2.loc[y, '%Var '+str(Su)] = round(((data2.loc[y, 'Prom '+str(Su)] - data2.loc[y, 'Prom NoOf']) / data2.loc[y, 'Prom NoOf']) * 100, 0)
                else:
                    continue
            else:
                data2.loc[y, 'Prom ' + str(Su)] = ''
                data2.loc[y, 'Cuenta ' + str(Su)] = ''
                data2.loc[y, '%Var ' + str(Su)] = ''

data2 = data2.fillna('')
data2 = data2.round(0)

# Se crea una 3er tabla donde se mantiene solo el análisis acumulado.
data3 = data2.drop(data2.iloc[: , 2: list(data2.columns).index('Cant Of')], axis=1 )  # data250.drop(data250.iloc[:, 2:557].columns, axis=1) # REEMPLAZADO PARA AUTOMATIZAR EL LENGTH.

print('75%')

clases = pd.read_excel('Nuevas Clases Elaborados.xlsx') # Para cualquier departamento que no sea Elaborados, acá hay que usar el plano.

data3.insert(3, 'Cuenta NoOf.', data3['Cuenta NoOf'], True)
data3.insert(4, 'Peso Of', ((data3['Cant Of']/(data3['Cuenta NoOf.']+data3['Cant Of']))*100).round(0))
data4 = data3.drop(['%Var NoOf', 'Cuenta Of', 'Cuenta NoOf'], axis=1)
data4.rename(columns={'Cuenta NoOf.':'Cuenta NoOf'}, inplace=True)
data4.fillna(0, inplace=True)
data4 = data4.replace('', 0)

for u, row in data4.iterrows():
    try:
        data4.loc[u, 'Clase'] = clases[clases['Producto']==row['Producto']]['Clase Descrip'].values[0]
    except:
        data4.loc[u, 'Clase'] = ''

por_clase = data4[data4['Prom Of']>0].groupby('Clase').sum()
por_clase = por_clase.drop(['Producto', 'Peso Of', '%Var Of', '%Var 17', '%Var 20', '%Var 25', '%Var 30', '%Var 33', '%Var 35', '%Var 40', '%Var 50','Prom 17','Prom 20','Prom 25','Prom 30','Prom 33','Prom 35','Prom 40','Prom 50'], axis=1)
por_clase.insert(2, '%Peso Of', round(por_clase['Cant Of']/(por_clase['Cant Of']+por_clase['Cuenta NoOf'])*100),0)
por_clase.insert(5, '%Var Of' ,round((por_clase['Prom Of'] - por_clase['Prom NoOf'])/por_clase['Prom NoOf']*100),0)
por_clase = por_clase.drop(['Cant Of', 'Cuenta NoOf', 'Prom NoOf', 'Prom Of'], axis=1)
por_clase.reset_index(inplace=True)

print('100% - Creando Excel')

data2.columns = data2.columns.astype('str') # Se pasa la fecha a String para poder splitear y buscar por aproximación.
semana_sgte = dt.now().isocalendar()[1] + 1
anopasado = dt.now().isocalendar()[0] - 1
fecha = str(dte.fromisocalendar(anopasado, semana_sgte, 1))
indice = list(data2.columns).index(fecha + ' 00:00:00')

portada = data2.iloc[:, [1, indice, indice + 1, indice + 2, indice + 3, indice + 4]][data2.iloc[:, indice:indice + 6].iloc[:, 1] == 'SI']
pivoted = portada.pivot_table(index= [list(portada.columns)[5], 'Descrip'], values=list(portada.columns), aggfunc='sum')

writer = pd.ExcelWriter('Análisis Venta COTO.xlsx', engine= 'xlsxwriter')
pivoted.to_excel(writer, index=True, sheet_name='Año pasado')
data2.to_excel(writer, index=False, sheet_name='Venta')
data4.to_excel(writer, index=False, sheet_name='Consolidado')
por_clase.to_excel(writer, index=False, sheet_name='Por Clase')

# Variables del xlsxwriter
workbook = writer.book
worksheet1 = writer.sheets['Venta']
worksheet2 = writer.sheets['Consolidado']
worksheet3 = writer.sheets['Por Clase']
worksheet4 = writer.sheets['Año pasado']

# Objetos de formato
percent_fmt = workbook.add_format({'num_format': '0.0%','border':1,'fg_color':'#DCE6F1'})
header_format = workbook.add_format({'bold':True, 'text_wrap':True, 'fg_color':'#71A3D9', 'border':2})
body_format = workbook.add_format({'border':1,'fg_color':'#DCE6F1'})
body_format2 = workbook.add_format({'border':1})
header_format.set_align('center')
header_format.set_align('vcenter')
body_format.set_align('center')
body_format.set_align('vcenter')
body_format2.set_align('center')
body_format2.set_align('vcenter')
percent_fmt.set_align('center')
percent_fmt.set_align('vcenter')

# Seteo de celdas, filas y columnas.
worksheet3.write_row(0, 0, list(por_clase.columns), header_format)
for i in range(len(por_clase)):
    worksheet3.write_row(i + 1, 0, por_clase.iloc[i, :], body_format)
for i in range(len(por_clase)):
    worksheet3.write_row(i + 1, 1, por_clase.iloc[i, 1:3] / 100, percent_fmt)

# worksheet4.write_row(0, 0, list(pivoted.columns), header_format)
# for i in range(len(pivoted)):
#     worksheet4.write_row(i + 1, 0, pivoted.iloc[i, :], body_format)

(max_row, max_col) = data2.shape
column_settings = [{'header':column} for column in data2.columns.astype('str')]
worksheet1.add_table(0,0, max_row, max_col-1, {'columns': column_settings})
# worksheet1.set_column(0, max_col, 12)

(max_row, max_col) = data4.shape
column_settings = [{'header':column} for column in data4.columns.astype('str')]
worksheet2.add_table(0,0, max_row, max_col-1, {'columns': column_settings})
# worksheet2.set_column(0, max_col, 12)

# Seteo de las páginas
worksheet3.set_column('B:K', 10)
worksheet3.set_column('A:A', 30)
# worksheet1.set_column('A:A',None, None,{'hidden':True})
worksheet1.set_column('B:B', 35)
worksheet1.set_column('C:XX', 15)
# worksheet2.set_column('A:A',None, None,{'hidden':True})
worksheet2.set_column('B:B', 35)
worksheet2.set_column('C:XX', 15)
# worksheet3.set_column('A:A',None, None, {'hidden':True})
worksheet4.set_column('B:B', 35)
worksheet4.set_column('A:A', 15)
worksheet4.set_column('C:E', 15)

print('OK')
writer.save()





