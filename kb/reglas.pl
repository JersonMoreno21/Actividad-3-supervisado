% ==========================================================================
% Reglas lógicas del sistema MIO — inteligencia artificial simbólica
% Archivo EXTERNO de reglas (el motor de inferencia lo parsea y aplica
% mediante backward chaining + unificación).
%
% Los HECHOS (estacion/4, conecta/4, etc.) se generan desde los datasets
% oficiales en data/*.csv (ver mio_router/builder.py).
% Fuente de datos: MetroCali — https://www.metrocali.gov.co
% ==========================================================================

% --- Conexión directa entre estaciones (un salto) ---
conectado(X, Y) :- conecta(X, Y, _, _).

% --- Viaje directo por una ruta/corredor específica ---
viaje_directo(X, Y, R) :- conecta(X, Y, R, _).

% --- Estación pertenece a un corredor/ruta ---
pertenece_corredor(E, R) :- sirve(R, E).

% --- Existe una ruta posible de un salto o esperando transbordo ---
posible_transbordo(E) :- transbordo(E, _, _, _).

% --- Alcanzable por un salto ---
alcanzable_un_salto(X, Y) :- conecta(X, Y, _, _).

% --- Alcanzable con hasta dos saltos (conexión intermedia Z) ---
alcanzable_dos_saltos(X, Y) :- conecta(X, Z, _, _), conecta(Z, Y, _, _).

% --- Misma zona de integración (ESTACION_M agrupa terminales/zonas) ---
misma_zona(X, Y) :- estacion(_, X, _, Z), estacion(_, Y, _, Z).

% --- Parada externa cerca de una estación ---
en_ultimate_milla(P, E) :- cerca_de(P, E, _).
