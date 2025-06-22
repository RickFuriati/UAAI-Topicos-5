import pandas as pd
import folium
import shapely as shp
import numpy as np
from pyproj import Transformer
from shapely import wkt
import os 
from geopy.distance import geodesic
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from folium.plugins import HeatMap
import streamlit as st
import plotly.express as px
from branca.element import Element

#data_path=r'C:\Users\rickg\iCloudDrive\Computer Science\P-08\Topicos 5\Data'
#base_path=r'C:\Users\rickg\iCloudDrive\Computer Science\P-08\Topicos 5'

base_path= os.getcwd()
data_path = os.path.join(base_path, 'Data')

#geometry=shp.wkt.loads(radar['GEOMETRIA'].iloc[0])

def find_coordinates(geometry):
    lats, lons = [], []
    transformer = Transformer.from_crs("EPSG:32723", "EPSG:4326", always_xy=True)
    for i, geom_str in enumerate(geometry):
        point = wkt.loads(geom_str)
        lon_utm, lat_utm = point.x, point.y
        lon, lat = transformer.transform(lon_utm, lat_utm)
        lats.append(lat)
        lons.append(lon)

    return lats, lons

def find_coordinates_accidents(lat_col,lon_col):
    lats, lons = [], []
    transformer = Transformer.from_crs("EPSG:32723", "EPSG:4326", always_xy=True)

    for lat, lon in zip(lat_col, lon_col):
        lon_utm, lat_utm = lon, lat
        lon, lat = transformer.transform(lon_utm, lat_utm)
        lats.append(lat)
        lons.append(lon)

    return lats, lons

def find_coordinates_line(geometry):
    lats, lons = [], []
    transformer = Transformer.from_crs("EPSG:32723", "EPSG:4326", always_xy=True)

    for geom_str in geometry:
        geom = wkt.loads(geom_str)

        # Converte para ponto dependendo do tipo
        if isinstance(geom, shp.Point):
            point = geom

        elif isinstance(geom, shp.LineString):
            point = geom.interpolate(0.5, normalized=True)  # ponto médio

        elif isinstance(geom, shp.MultiLineString):
            largest = max(geom.geoms, key=lambda g: g.length)  # maior segmento
            point = largest.interpolate(0.5, normalized=True)

        else:
            raise ValueError(f"Tipo de geometria não suportado: {type(geom)}")

        # Projeta para lat/lon
        lon_utm, lat_utm = point.x, point.y
        lon, lat = transformer.transform(lon_utm, lat_utm)

        lats.append(lat)
        lons.append(lon)

    return lats, lons

# Converte graus para radianos
def to_radians(df):
    return np.radians(df[['LATITUDE', 'LONGITUDE']].values)

# Função genérica para contar vizinhos em raio
def contar_vizinhos_balltree(pontos_referencia, pontos_alvo, raio_km):
    tree = BallTree(to_radians(pontos_alvo), metric='haversine')
    raio_rad = raio_km / 6371.0  # raio da Terra em km
    counts = tree.query_radius(to_radians(pontos_referencia), r=raio_rad, count_only=True)
    return counts

def generate_cluster_map():
    centro=[-19.9227417, -43.9451139]


    radar= pd.read_csv(os.path.join(data_path, '20250602_fiscalizacao_eletronica.csv'), sep=';', encoding='latin1')

    lats, longs = find_coordinates(radar['GEOMETRIA'])
    radar['LATITUDE'] = lats
    radar['LONGITUDE'] = longs

    semaforo = pd.read_csv(os.path.join(data_path, '20250602_sinalizacao_semaforica.csv'), sep=';', encoding='latin1')
    lats, longs = find_coordinates(semaforo['GEOMETRIA'])
    semaforo['LATITUDE'] = lats
    semaforo['LONGITUDE'] = longs

    redutor=pd.read_csv(os.path.join(data_path, '20250602_redutor_velocidade.csv'), sep=';', encoding='latin1')
    lats, longs = find_coordinates_line(redutor['GEOMETRIA'])
    redutor['LATITUDE'] = lats
    redutor['LONGITUDE'] = longs

    distancias = np.array([
        geodesic((lat, lon), (centro[0], centro[1])).km
        for lat, lon in zip(lats, longs)
    ])

    dentro_raio = distancias <= 10
    redutor = redutor[dentro_raio]

    accidents= pd.read_csv(os.path.join(data_path, 'si-bol-2023.csv'), sep=';', encoding='latin1')
    accidents = accidents[accidents[' COORDENADA_X'] !=0]
    accidents = accidents[accidents[' COORDENADA_Y'] !=0]
    lats,longs = find_coordinates_accidents(accidents[' COORDENADA_Y'], accidents[' COORDENADA_X'])

    accidents['LATITUDE'] = lats
    accidents['LONGITUDE'] = longs

    accidents['N_SEMAFOROS_1KM'] = contar_vizinhos_balltree(accidents, semaforo, 0.4)
    accidents['N_LOMBADAS_1KM']  = contar_vizinhos_balltree(accidents, redutor, 0.4)
    accidents['N_RADARES_1KM']   = contar_vizinhos_balltree(accidents, radar, 0.4)

    accidents['TIPO_ACIDENTE_COD'] = LabelEncoder().fit_transform(accidents[' DESC_TIPO_ACIDENTE'])

    X= accidents[['LATITUDE', 'LONGITUDE','TIPO_ACIDENTE_COD','N_SEMAFOROS_1KM','N_LOMBADAS_1KM','N_RADARES_1KM']].values


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    accidents['CLUSTER'] = kmeans.fit_predict(X_scaled)

    #clustering = DBSCAN(eps=0.1, min_samples=10).fit(X)
    #accidents['CLUSTER'] = clustering.labels_

    accidents_outliers = accidents[accidents['CLUSTER'] ==-1]
    accidents = accidents[accidents['CLUSTER'] !=-1]


    n_clusters = accidents['CLUSTER'].nunique()
    cores = plt.cm.get_cmap('tab20', n_clusters)

    cluster_labels = sorted(accidents['CLUSTER'].unique())
    cor_por_cluster = {
        cluster: rgb2hex(cores(i)[:3])
        for i, cluster in enumerate(cluster_labels)
    }


    m = folium.Map(
        location=centro,
        zoom_start=14,
        tiles=None  
    )

    radar_group = folium.FeatureGroup(name='Radar',show=False)
    semaforo_group = folium.FeatureGroup(name='Semáforo',show=False)
    lombada_group = folium.FeatureGroup(name='Lombada',show=False)
    cluster_group = folium.FeatureGroup(name='Clusters de Acidentes')

    folium.TileLayer(
        tiles='https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png',
        attr='© OpenStreetMap contributors, © CARTO',
        name='CartoDB Positron No Labels',
        control=False,
        overlay=False
    ).add_to(m)

    for lat, lon in zip(semaforo['LATITUDE'], semaforo['LONGITUDE']):
        custom_icon = folium.CustomIcon(
            icon_image=base_path + '\\Icons\\luzes-de-transito.png',  # Certifique-se de que o caminho do ícone está correto
            icon_size=(15, 15)
        )
        folium.Marker(
            location=[lat, lon],
            popup="Semáforo",
            icon=custom_icon
        ).add_to(semaforo_group)

    for lat, lon in zip(redutor['LATITUDE'], redutor['LONGITUDE']):
        custom_icon = folium.CustomIcon(
            icon_image=base_path + '\\Icons\\bump.png',  # Certifique-se de que o caminho do ícone está correto
            icon_size=(15,15)
        )
        folium.Marker(
            location=[lat, lon],
            popup="Redutor de Velocidade",
            icon=custom_icon
        ).add_to(lombada_group)

    for lat, lon, vel in zip(radar['LATITUDE'], radar['LONGITUDE'], radar['VELOCIDADE_REGULAMENTAR']):
        custom_icon = folium.CustomIcon(
            icon_image=base_path + '\\Icons\\car-and-radar-security.png',  # Certifique-se de que o caminho do ícone está correto
            icon_size=(15, 15)
        )
        folium.Marker(
            location=[lat, lon],
            popup="Velocidade Permitida: " + str(vel),
            icon=custom_icon
        ).add_to(radar_group)



    for _, row in accidents.iterrows():
        cluster = row['CLUSTER']
        cor = cor_por_cluster[cluster]
        
        folium.CircleMarker(
            location=[row['LATITUDE'], row['LONGITUDE']],
            radius=1,
            color=cor,
            fill=True,
            fill_color=cor,
            fill_opacity=0.8,
            popup=f"Tipo: {row[' DESC_TIPO_ACIDENTE']}<br>Cluster: {cluster}<br>Semáforos em 400m: {row['N_SEMAFOROS_1KM']}<br>Lombadas em 400m: {row['N_LOMBADAS_1KM']}<br>Radares em 400m: {row['N_RADARES_1KM']}"
        ).add_to(cluster_group)


    radar_group.add_to(m)
    semaforo_group.add_to(m)
    lombada_group.add_to(m)
    cluster_group.add_to(m)


    folium.LayerControl(collapsed=False).add_to(m)

    legenda_linhas = "\n".join([
        f'<i style="color:{cor};">■</i> Cluster {cluster}<br>'
        for cluster, cor in cor_por_cluster.items()
    ])
    import textwrap
    
    legend_html = textwrap.dedent(f"""
        <div style="
            position: absolute;
            top: 150px;  /* desloca para baixo do LayerControl */
            right: 10px;
            width: 200px;
            background-color: white;
            z-index:9999;
            font-size: 14px;
            border: 2px solid grey;
            border-radius: 6px;
            padding: 10px;">
            <b>Legenda por Cluster</b><br>
            {''.join([f'<i style="color:{cor};">■</i> Cluster {cluster}<br>' for cluster, cor in cor_por_cluster.items()])}
        </div>
    """)

    m.get_root().html.add_child(Element(legend_html))

    # Salva o mapa
    m.save(str(base_path) + '\\clusters.html')

def generate_heatmap():
    centro=[-19.9227417, -43.9451139]

    #accidents_01= pd.read_csv(data_path+'\\20250602_sinistro_transito_vitima.csv',sep=';', encoding='latin1')

    radar= pd.read_csv(os.path.join(data_path, '20250602_fiscalizacao_eletronica.csv'), sep=';', encoding='latin1')
    lats, longs = find_coordinates(radar['GEOMETRIA'])
    radar['LATITUDE'] = lats
    radar['LONGITUDE'] = longs

    semaforo = pd.read_csv(os.path.join(data_path, '20250602_sinalizacao_semaforica.csv'), sep=';', encoding='latin1')
    lats, longs = find_coordinates(semaforo['GEOMETRIA'])
    semaforo['LATITUDE'] = lats
    semaforo['LONGITUDE'] = longs

    redutor=pd.read_csv(os.path.join(data_path, '20250602_redutor_velocidade.csv'), sep=';', encoding='latin1')
    lats, longs = find_coordinates_line(redutor['GEOMETRIA'])
    redutor['LATITUDE'] = lats
    redutor['LONGITUDE'] = longs

    distancias = np.array([
        geodesic((lat, lon), (centro[0], centro[1])).km
        for lat, lon in zip(lats, longs)
    ])

    dentro_raio = distancias <= 10
    redutor = redutor[dentro_raio]

    accidents= pd.read_csv(os.path.join(data_path, 'si-bol-2023.csv'), sep=';', encoding='latin1')
    accidents = accidents[accidents[' COORDENADA_X'] !=0]
    accidents = accidents[accidents[' COORDENADA_Y'] !=0]
    lats,longs = find_coordinates_accidents(accidents[' COORDENADA_Y'], accidents[' COORDENADA_X'])

    accidents['LATITUDE'] = lats
    accidents['LONGITUDE'] = longs

    heat_data = accidents[['LATITUDE', 'LONGITUDE']].dropna().values.tolist()

    m = folium.Map(

        location=centro,
        zoom_start=16,
        tiles=None  
    )

    radar_group = folium.FeatureGroup(name='Radar',show=False)
    semaforo_group = folium.FeatureGroup(name='Semáforo',show=False)
    lombada_group = folium.FeatureGroup(name='Lombada',show=False)
    accidents_group = folium.FeatureGroup(name='Acidentes',show=False)
    heatmap_layer = folium.FeatureGroup(name='Densidade de Acidentes')

    HeatMap(
        heat_data,
        radius=25,
        blur=30,
        min_opacity=0.7  # valor entre 0 (transparente) e 1 (opaco)
    ).add_to(heatmap_layer)

    folium.TileLayer(
        tiles='https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png',
        attr='© OpenStreetMap contributors, © CARTO',
        name='CartoDB Positron No Labels',
        control=False,
        overlay=False
    ).add_to(m)

    heatmap_layer.add_to(m)

    for lat, lon in zip(semaforo['LATITUDE'], semaforo['LONGITUDE']):
        custom_icon = folium.CustomIcon(
            icon_image=base_path + '\\Icons\\luzes-de-transito.png',  # Certifique-se de que o caminho do ícone está correto
            icon_size=(15, 15)
        )
        folium.Marker(
            location=[lat, lon],
            popup="Semáforo",
            icon=custom_icon
        ).add_to(semaforo_group)

    for lat, lon in zip(redutor['LATITUDE'], redutor['LONGITUDE']):
        custom_icon = folium.CustomIcon(
            icon_image=base_path + '\\Icons\\bump.png',  # Certifique-se de que o caminho do ícone está correto
            icon_size=(25,25)
        )
        folium.Marker(
            location=[lat, lon],
            popup="Redutor de Velocidade",
            icon=custom_icon
        ).add_to(lombada_group)

    for lat, lon, vel in zip(radar['LATITUDE'], radar['LONGITUDE'], radar['VELOCIDADE_REGULAMENTAR']):
        custom_icon = folium.CustomIcon(
            icon_image=base_path + '\\Icons\\car-and-radar-security.png',  # Certifique-se de que o caminho do ícone está correto
            icon_size=(25, 25)
        )
        folium.Marker(
            location=[lat, lon],
            popup="Velocidade Permitida: " + str(vel),
            icon=custom_icon
        ).add_to(radar_group)

    for lat, lon,date, desc in zip(accidents['LATITUDE'], accidents['LONGITUDE'], accidents[' DATA HORA_BOLETIM'],accidents[' DESC_TIPO_ACIDENTE']):
        custom_icon = folium.CustomIcon(
            icon_image=base_path + '\\Icons\\warning.png',  # Certifique-se de que o caminho do ícone está correto
            icon_size=(15, 15)
        )
        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.8,
            popup=f"Data: {date}<br>Descrição: {desc}",
            icon=custom_icon
        ).add_to(accidents_group)


    radar_group.add_to(m)
    semaforo_group.add_to(m)
    lombada_group.add_to(m)
    accidents_group.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # Salva o mapa
    m.save(str(base_path) + '\\heatmap.html')

def generate_accidents_map():
    centro=[-19.9227417, -43.9451139]

    radar= pd.read_csv(data_path+'\\20250602_fiscalizacao_eletronica.csv',sep=';', encoding='latin1')
    lats, longs = find_coordinates(radar['GEOMETRIA'])
    radar['LATITUDE'] = lats
    radar['LONGITUDE'] = longs

    semaforo = pd.read_csv(data_path+'\\20250602_sinalizacao_semaforica.csv',sep=';', encoding='latin1')
    lats, longs = find_coordinates(semaforo['GEOMETRIA'])
    semaforo['LATITUDE'] = lats
    semaforo['LONGITUDE'] = longs

    redutor=pd.read_csv(data_path+'\\20250602_redutor_velocidade.csv',sep=';', encoding='latin1')
    lats, longs = find_coordinates_line(redutor['GEOMETRIA'])
    redutor['LATITUDE'] = lats
    redutor['LONGITUDE'] = longs

    distancias = np.array([
        geodesic((lat, lon), (centro[0], centro[1])).km
        for lat, lon in zip(lats, longs)
    ])

    dentro_raio = distancias <= 10
    redutor = redutor[dentro_raio]

    accidents= pd.read_csv(data_path+'\\si-bol-2023.csv',sep=';', encoding='latin1')
    accidents = accidents[accidents[' COORDENADA_X'] !=0]
    accidents = accidents[accidents[' COORDENADA_Y'] !=0]
    lats,longs = find_coordinates_accidents(accidents[' COORDENADA_Y'], accidents[' COORDENADA_X'])

    accidents['LATITUDE'] = lats
    accidents['LONGITUDE'] = longs

    accidents['N_SEMAFOROS_1KM'] = contar_vizinhos_balltree(accidents, semaforo, 0.4)
    accidents['N_LOMBADAS_1KM']  = contar_vizinhos_balltree(accidents, redutor, 0.4)
    accidents['N_RADARES_1KM']   = contar_vizinhos_balltree(accidents, radar, 0.4)

    m = folium.Map(
        location=centro,
        zoom_start=16,
        tiles=None  
    )

    radar_group = folium.FeatureGroup(name='Radar',show=False)
    semaforo_group = folium.FeatureGroup(name='Semáforo',show=False)
    lombada_group = folium.FeatureGroup(name='Lombada',show=False)
    accidents_group = folium.FeatureGroup(name='Acidentes',show=True)

    folium.TileLayer(
        tiles='https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png',
        attr='© OpenStreetMap contributors, © CARTO',
        name='CartoDB Positron No Labels',
        control=False,
        overlay=False
    ).add_to(m)

    for lat, lon in zip(semaforo['LATITUDE'], semaforo['LONGITUDE']):
        custom_icon = folium.CustomIcon(
            icon_image=base_path + '\\Icons\\luzes-de-transito.png',  # Certifique-se de que o caminho do ícone está correto
            icon_size=(15, 15)
        )
        folium.Marker(
            location=[lat, lon],
            popup="Semáforo",
            icon=custom_icon
        ).add_to(semaforo_group)

    for lat, lon in zip(redutor['LATITUDE'], redutor['LONGITUDE']):
        custom_icon = folium.CustomIcon(
            icon_image=base_path + '\\Icons\\bump.png',  # Certifique-se de que o caminho do ícone está correto
            icon_size=(25,25)
        )
        folium.Marker(
            location=[lat, lon],
            popup="Redutor de Velocidade",
            icon=custom_icon
        ).add_to(lombada_group)

    for lat, lon, vel in zip(radar['LATITUDE'], radar['LONGITUDE'], radar['VELOCIDADE_REGULAMENTAR']):
        custom_icon = folium.CustomIcon(
            icon_image=base_path + '\\Icons\\car-and-radar-security.png',  # Certifique-se de que o caminho do ícone está correto
            icon_size=(25, 25)
        )
        folium.Marker(
            location=[lat, lon],
            popup="Velocidade Permitida: " + str(vel),
            icon=custom_icon
        ).add_to(radar_group)

    for lat, lon,date, desc in zip(accidents['LATITUDE'], accidents['LONGITUDE'], accidents[' DATA HORA_BOLETIM'],accidents[' DESC_TIPO_ACIDENTE']):
        custom_icon = folium.CustomIcon(
            icon_image=base_path + '\\Icons\\warning.png',  # Certifique-se de que o caminho do ícone está correto
            icon_size=(15, 15)
        )
        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.8,
            popup=f"Data: {date}<br>Descrição: {desc}",
            icon=custom_icon
        ).add_to(accidents_group)

    radar_group.add_to(m)
    semaforo_group.add_to(m)
    lombada_group.add_to(m)
    accidents_group.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # Salva o mapa
    m.save(str(base_path) + '\\accidents.html')

mapa_clusters_path = 'clusters.html'
mapa_heatmap_path = 'heatmap.html'
mapa_accidents_path = 'accidents.html'

# Carregar os HTMLs
with open(mapa_clusters_path, 'r', encoding='utf-8') as f:
    mapa_clusters_html = f.read()

with open(mapa_heatmap_path, 'r', encoding='utf-8') as f:
    mapa_heatmap_html = f.read()

with open(mapa_accidents_path, 'r', encoding='utf-8') as f:
    mapa_accidents_html = f.read()

st.set_page_config(layout="wide")
st.markdown("<style>section.main > div {padding-top: 0rem;}</style>", unsafe_allow_html=True)

col1, col2 = st.columns([3, 2])
st.title("Utilitário de Análise de Acidentes Inteligente (UAAI)")

with st.container():
    # Layout de colunas (mapa mais largo que o gráfico)
    col1, col2 = st.columns([4, 2])  # proporção 4:2 para mapa ficar dominante
    with col1:
        # Seletor sobre o mapa
        map_type = st.selectbox("Tipo de mapa:", ["Clusters", "Heatmap", "Acidentes - 2023"])

        # Mapa ocupa 100% da largura da coluna
        if map_type == "Clusters":
            st.components.v1.html(mapa_clusters_html, height=800, width=None, scrolling=False)
        elif map_type == "Heatmap":
            st.components.v1.html(mapa_heatmap_html, height=800, width=None, scrolling=False)
        elif map_type == "Acidentes - 2023":
            st.components.v1.html(mapa_accidents_html, height=800, width=None, scrolling=False)

with col2:
    st.markdown("### Acidentes ao longo do tempo")
    accidents= pd.read_csv(os.path.join(data_path, '20250602_sinistro_transito_vitima.csv'), sep=';', encoding='latin1')
    #columns=accidents_01.columns
    #df_time = accidents_01.groupby('DATA_HORA_BOLETIM').sum().reset_index()

    accidents['DATA'] = pd.to_datetime(accidents['DATA_HORA_BOLETIM'])
    accidents['MES'] = accidents['DATA'].dt.to_period('M')
    df_mensal = accidents.groupby('MES').size().reset_index(name='Quantidade')
    df_mensal['MES'] = df_mensal['MES'].dt.to_timestamp()

    st.line_chart(df_mensal.rename(columns={'MES': 'index'}).set_index('index'), use_container_width=True)   

    accidents= pd.read_csv(os.path.join(data_path, 'si-bol-2023.csv'), sep=';', encoding='latin1')
    accidents = accidents[accidents[' COORDENADA_X'] !=0] 

    df_accidents = accidents[' DESC_TIPO_ACIDENTE'].value_counts().reset_index()
    df_accidents.columns = ['Tipo de Acidente', 'Quantidade']
    df_accidents = df_accidents[df_accidents['Quantidade'] > 0]
    #df_accidents=df_accidents.set_index('Tipo de Acidente')

    fig = px.pie(df_accidents, values='Quantidade', names='Tipo de Acidente',
            #title='Tipo de Acidente',
            color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_layout(showlegend=False)

    st.markdown("### Acidentes por Tipo (2023)")
    #st.bar_chart(df_accidents, use_container_width=True)
    st.plotly_chart(fig,use_container_width=True)
