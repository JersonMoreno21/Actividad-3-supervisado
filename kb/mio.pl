% Base de conocimiento del MIO de Santiago de Cali
% Datos ilustrativos y aproximados para actividades académicas.
% Fuente oficial: https://www.metrocali.gov.co

% Hechos de estaciones: estacion(ID, Nombre, Tipo, Zona)
estacion(1, paso_del_comercio, terminal, norte).
estacion(2, andres_sinan, terminal, norte).
estacion(3, calafate, estacion, norte).
estacion(4, el_techo, estacion, norte).
estacion(5, chipichape, estacion, norte).
estacion(6, cauca, estacion, norte).
estacion(7, malpelo, estacion, norte).
estacion(8, cañaveralejo, terminal, centro).
estacion(9, san_jose, estacion, centro).
estacion(10, plaza_de_mercado, estacion, centro).
estacion(11, 7_de_agosto, estacion, centro).
estacion(12, el_muelle, estacion, centro).
estacion(13, menga, terminal, sur).
estacion(14, calipso, estacion, sur).
estacion(15, universidades, terminal, sur).
estacion(16, las_celdas, estacion, sur).
estacion(17, mirador, estacion, sur).
estacion(18, km_15, parada, norte).
estacion(19, km_10, parada, norte).
estacion(20, km_5, parada, centro).

% Hechos de rutas: ruta(Codigo, Nombre, Tipo)
ruta(1, troncal_1, troncal).
ruta(2, troncal_2, troncal).
ruta(3, troncal_3, troncal).
ruta(4, pretroncal_A, pretroncal).
ruta(5, pretroncal_B, pretroncal).
ruta(6, alimentadora_1, alimentadora).
ruta(7, alimentadora_2, alimentadora).

% Sirve: sirve(Codigo_ruta, Estacion)
sirve(1, paso_del_comercio).
sirve(1, andres_sinan).
sirve(1, calafate).
sirve(1, el_techo).
sirve(1, chipichape).
sirve(1, cauca).
sirve(1, malpelo).
sirve(1, cañaveralejo).
sirve(1, san_jose).
sirve(1, plaza_de_mercado).
sirve(1, 7_de_agosto).
sirve(1, el_muelle).
sirve(1, menga).
sirve(1, calipso).
sirve(1, universidades).

% Conecta: Conecta(Origen, Destino, Ruta, Minutos)
conecta(paso_del_comercio, andres_sinan, 1, 4).
conecta(andres_sinan, calafate, 1, 3).
conecta(calafate, el_techo, 1, 3).
conecta(el_techo, chipichape, 1, 2).
conecta(chipichape, cauca, 1, 3).
conecta(cauca, malpelo, 1, 3).
conecta(malpelo, cañaveralejo, 1, 5).

conecta(cañaveralejo, san_jose, 1, 3).
conecta(san_jose, plaza_de_mercado, 1, 4).
conecta(plaza_de_mercado, 7_de_agosto, 1, 3).
conecta(7_de_agosto, el_muelle, 1, 3).

conecta(el_muelle, menga, 1, 4).
conecta(menga, calipso, 1, 3).
conecta(calipso, universidades, 1, 3).

% Rutas adicionales (troncal 2 y 3)
% Ruta 2: alternativa o circuito
conecta(paso_del_comercio, el_muelle, 2, 6).
conecta(el_muelle, universidades, 2, 5).

% Ruta 3: otro circuito
conecta(andres_sinan, san_jose, 3, 3).
conecta(san_jose, calipso, 3, 3).

% Pretroncales
conecta(calafate, km_15, 4, 2).
conecta(km_15, km_10, 4, 2).
conecta(km_10, km_5, 4, 2).
conecta(km_5, san_jose, 4, 3).

% Alimentadoras
conecta(paso_del_comercio, km_15, 6, 5).
conecta(andres_sinan, km_5, 7, 4).

% Ruta 5: sur
conecta(menga, mirador, 5, 2).
conecta(mirador, las_celdas, 5, 2).
conecta(las_celdas, universidades, 5, 3).

% Transbordos: estacion, ruta1, ruta2, minutos_espera
transbordo(san_jose, 1, 4, 5).
transbordo(calipso, 1, 5, 5).
transbordo(malpelo, 1, 2, 5).
transbordo(plaza_de_mercado, 1, 3, 5).
transbordo(7_de_agosto, 2, 3, 5).
transbordo(km_5, 4, 1, 3).

% Coordenadas: estacion(ID, Lat, Lon) - grados aproximados
coordenadas(paso_del_comercio, 3.4521, -76.5240).
coordenadas(andres_sinan, 3.4485, -76.5280).
coordenadas(calafate, 3.4450, -76.5320).
coordenadas(el_techo, 3.4415, -76.5360).
coordenadas(chipichape, 3.4380, -76.5400).
coordenadas(cauca, 3.4345, -76.5440).
coordenadas(malpelo, 3.4310, -76.5480).
coordenadas(cañaveralejo, 3.4530, -76.5150).
coordenadas(san_jose, 3.4560, -76.5100).
coordenadas(plaza_de_mercado, 3.4580, -76.5050).
coordenadas(7_de_agosto, 3.4600, -76.5000).
coordenadas(el_muelle, 3.4620, -76.4950).
coordenadas(menga, 3.4700, -76.4800).
coordenadas(calipso, 3.4720, -76.4750).
coordenadas(universidades, 3.4750, -76.4700).
coordenadas(las_celdas, 3.4800, -76.4650).
coordenadas(mirador, 3.4820, -76.4600).
coordenadas(km_15, 3.4400, -76.5300).
coordenadas(km_10, 3.4420, -76.5280).
coordenadas(km_5, 3.4450, -76.5250).