import streamlit as st
import pandas as pd
import time
import matplotlib.pyplot as plt
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from model import generate_network, infection_input, SIS

st.set_page_config(layout="wide", page_title="")

st.title("🦠 Simulatore di Epidemie Modello (SIS)")
st.markdown("""
Benvenuti! In questa app esploreremo come un'epidemia si diffonde all'interno di una popolazione e come le connessioni tra le persone (la "Rete") e le misure preventive (come il "Lockdown") influenzano il numero di contagiati.

""")

st.sidebar.header("⚙️ Impostazioni del Modello")

with st.sidebar.expander("🌐 1. Blocco Rete (Contatti Sociali)", expanded=True):
    st.markdown("**Come sono collegate le persone tra loro?**")
    n_nodes = st.number_input("Numero di Nodi (Popolazione Totale)", min_value=20, max_value=50000, value=10000, step=1000, help="Quante persone fanno parte della nostra città virtuale.")
    net_type = st.selectbox("Modello di Rete", ["Barabási-Albert", "Erdős-Rényi", "Small World"], index=1, help="Definisce come le persone si conoscono. 'Erdős-Rényi' è completamente casuale. 'Barabási-Albert' ha degli 'influencer' molto connessi. 'Small World' simula cerchie di amici chiuse con qualche conoscenza esterna.")
    
    if net_type == "Erdős-Rényi":
        p_erdos = st.number_input("Probabilità di connessione (p)", min_value=0.00001, max_value=1.0, value=0.005, step=0.001, format="%.4f", help="Probabilità matematica che due persone a caso si conoscano.")
        net_kwargs = {'p': p_erdos}
    elif net_type == "Barabási-Albert":
        m_barabasi = st.number_input("Numero di link iniziali (m)", min_value=1, max_value=100, value=25, step=1, help="Quante nuove connessioni fa una nuova persona quando entra nella rete. Numeri più alti significano più contatti.")
        net_kwargs = {'m': m_barabasi}
    elif net_type == "Small World":
        k_small = st.number_input("Grado iniziale (k)", min_value=2, max_value=100, value=50, step=2, help="Quanti 'vicini di casa' conosce ogni singola persona.")
        p_small = st.number_input("Probabilità di rewiring (p)", min_value=0.0, max_value=1.0, value=0.1, step=0.05, help="Probabilità che una persona conosca qualcuno fuori dal suo gruppo ristretto (es. amici online).")
        net_kwargs = {'k': k_small, 'p': p_small}

with st.sidebar.expander("🦠 2. Proprietà Epidemiologiche", expanded=False):
    st.markdown("**Caratteristiche del Virus:**")
    beta_min = st.number_input("Infettività Minima Estiva (β min)", min_value=0.0001, max_value=0.1, value=0.0005, step=0.0001, format="%.5f", help="La probabilità minima di trasmettere il virus (es. in estate quando il virus circola meno).")
    beta_max = st.number_input("Infettività Massima Invernale (β max)", min_value=0.0001, max_value=0.1, value=0.005, step=0.0001, format="%.5f", help="La probabilità massima di trasmettere il virus (es. in inverno, al chiuso).")
    gamma = st.number_input("Tasso Guarigione (γ)", min_value=0.01, max_value=1.0, value=0.07, step=0.01, help="Più questo valore è alto, più velocemente una persona infetta guarisce diventando di nuovo sana (e suscettibile).")
    death_rate = st.number_input("Tasso di Mortalità (μ)", min_value=0.00001, max_value=0.01, value=0.0001, step=0.0001, format="%.5f", help="Probabilità giornaliera che una persona infetta muoia.")
    infection_seed = st.number_input("Probabilità Infetti Iniziali", min_value=0.001, max_value=1.0, value=0.005, step=0.001, format="%.4f", help="All'inizio (giorno 0), quale percentuale di persone è già malata?")

with st.sidebar.expander("🛡️ 3. Blocco Lockdown", expanded=False):
    st.markdown("**Come difendersi?**")
    lockdown_threshold = st.slider("Soglia di Allarme per Lockdown (%)", min_value=1, max_value=100, value=20, help="Il governo fa scattare il lockdown quando questa percentuale della popolazione di ammala contemporaneamente.") / 100
    lockdown_duration = st.number_input("Durata Lockdown (giorni)", min_value=7, max_value=365, value=60, step=1, help="Per quanti giorni consecutivi manteniamo le restrizioni prima di riaprire.")
    lock_eff_percent = st.slider("Efficacia: Percentuale di link sociali tagliati (%)", min_value=0, max_value=100, value=40, help="Se vale 40%, significa che durante il lockdown il 40% degli incontri sociali viene evitato (es. scuole e uffici chiusi).")
    lock_eff = lock_eff_percent / 100

st.sidebar.markdown("---")
max_steps = st.sidebar.number_input("Giorni di Simulazione (max step)", min_value=100, max_value=2000, value=730, step=30, help="Quanti giorni di epidemia vuoi simulare?")
enable_animation = st.sidebar.checkbox("Vuoi vedere la riproduzione in tempo reale (animata)?", value=False, help="Se spuntato, vedrai il grafico costruirsi giorno dopo giorno.")
enable_net_animation = st.sidebar.checkbox("Vuoi vedere l'evoluzione dell'epidemia sulla Rete ?", value=False, help="Mostra i pallini rossi e blu. Sconsigliato per reti grandi, va bene se Nodi < 1000.")

speed_msg = st.sidebar.empty()

if enable_animation or enable_net_animation:
    speed = st.sidebar.slider("Passo dei giorni (velocità animazione)", min_value=1, max_value=50, value=5, help="1 significa che l'animazione mostra i giorni uno a uno. Più è alto, più veloce va ma 'salta' i giorni intermedi.")
else:
    speed = 1

if st.sidebar.button("🚀 Avvia Simulazione", type="primary"):
    with st.spinner("Generazione della rete in corso (può impiegare qualche secondo)..."):
        adj_list, G = generate_network(n_nodes, net_type, **net_kwargs)
        degrees = [d for n, d in G.degree()]
        avg_degree = sum(degrees) / n_nodes if n_nodes > 0 else 0
        
    st.success(f"Rete {net_type} generata! Nodi: {n_nodes}, Grado Medio delle connessioni: {avg_degree:.2f}")
    
    with st.spinner("Simulazione in corso..."):
        infection_dict = infection_input(infection_seed, n_nodes)
        
        # Esecuzione del modello SIS in background
        infected_over_time, daily_infected_arr, daily_deaths_arr, cumulative_deaths_arr, lockdown_status_arr, node_states_list = SIS(
            adj_list, infection_dict, beta_min, beta_max, gamma, death_rate, 
            lock_eff, lockdown_threshold, lockdown_duration, max_steps
        )
        
        # Preparazione Dati finali
        df = pd.DataFrame({
            'Giorno': range(max_steps),
            'Infetti Attuali': infected_over_time,
            'Nuovi Infetti': daily_infected_arr,
            'Decessi Cumulativi': cumulative_deaths_arr,
            'Lockdown Attivo': lockdown_status_arr
        })
        
        # Calcolo approssimativo di Rt
        # Evitiamo pd.NA esplicito per non far crashare .rolling()
        df['Infetti Precedenti'] = df['Infetti Attuali'].shift(1).replace(0, float('nan'))
        df['Rt (Grezzo)'] = df['Nuovi Infetti'] / (df['Infetti Precedenti'] * gamma)
        df['Rt (Grezzo)'] = df['Rt (Grezzo)'].fillna(0).astype(float)
        
        # Media mobile a 7 giorni
        df['R_t'] = df['Rt (Grezzo)'].rolling(window=7, min_periods=1).mean().fillna(0)
        
    st.markdown("### 📊 Risultati Simulazione in Tempo Reale")
    
    status_placeholder = st.empty()
    
    # Creiamo tre colonne e mettiamo la rete solo al centro per rimpicciolirne l'ingombro visivo
    col_vuota1, col_rete, col_vuota2 = st.columns([1, 2, 1])
    network_placeholder = col_rete.empty()
    
    pos = None
    if enable_net_animation:
        # Per evitare crash o attese infinite su reti grosse
        if n_nodes > 2000:
            st.warning(f"Attenzione: hai chiesto l'animazione della rete su {n_nodes} nodi. La visualizzazione potrebbe essere lenta o bloccarsi. È consigliato usare N < 2000 per questa funzione.")
            
        with st.spinner("Calcolo disposizione dei nodi per l'animazione (solo la prima volta)..."):
            pos = nx.spring_layout(G, seed=42)
            
    col1, col2 = st.columns(2)
    chart_infetti = col1.empty()
    chart_nuovi = col2.empty()
    
    col3, col4 = st.columns(2)
    chart_morti = col3.empty()
    chart_rt = col4.empty()
    
    max_infected_ever = int(df['Infetti Attuali'].max())
    y_max_infetti = max_infected_ever + (max_infected_ever * 0.1) if max_infected_ever > 0 else 10
    
    max_nuovi_ever = int(df['Nuovi Infetti'].max())
    y_max_nuovi = max_nuovi_ever + (max_nuovi_ever * 0.1) if max_nuovi_ever > 0 else 10
    
    max_morti_ever = int(df['Decessi Cumulativi'].max())
    y_max_morti = max_morti_ever + (max_morti_ever * 0.1) if max_morti_ever > 0 else 10
    
    max_rt_ever = df['R_t'].max()
    y_max_rt = max_rt_ever + (max_rt_ever * 0.1) if not pd.isna(max_rt_ever) and max_rt_ever > 0 else 3.0
    
    if enable_animation or enable_net_animation:
        # Animazione per fare vedere la curva che si espande e/o la rete
        for i in range(1, max_steps + 1, speed):
            if i > max_steps:
                 i = max_steps
                 
            sub_df = df.iloc[:i]
            
            # --- AGGIORNAMENTO GRAFICO DELLA RETE ---
            if enable_net_animation and pos is not None:
                # Dimensione del grafo rimpicciolita
                fig_net, ax_net = plt.subplots(figsize=(3, 3))
                
                current_states = node_states_list[i-1]
                colors = ['red' if state == 1 else 'blue' for state in current_states]
                
                # Disegna archi leggermente trasparenti
                nx.draw_networkx_edges(G, pos, ax=ax_net, alpha=0.1)
                # Disegna nodi col loro stato al momento i
                nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=15, ax=ax_net)
                
                ax_net.set_title(f"Diffusione Virus sulla Rete - Giorno {i}")
                ax_net.axis('off')
                
                # Forza il grafico a non occupare tutta la larghezza container se non in quella colonna
                network_placeholder.pyplot(fig_net, use_container_width=True, clear_figure=True)
                plt.close(fig_net)
            
            if enable_animation:
                # 1. Plot Infetti Attuali
                fig1, ax1 = plt.subplots(figsize=(5, 3))
                ax1.plot(sub_df['Giorno'], sub_df['Infetti Attuali'], color='red', label='Infetti', linewidth=2)
                lockdown_days = sub_df[sub_df['Lockdown Attivo'] == 1]
                if not lockdown_days.empty:
                    ax1.scatter(lockdown_days['Giorno'], lockdown_days['Infetti Attuali'], color='orange', label='Lockdown', marker='s', s=10)
                ax1.set_title("Progressione Infetti Attuali")
                ax1.set_xlim(0, max_steps)
                ax1.set_ylim(0, y_max_infetti)
                ax1.grid(alpha=0.3)
                handles, labels = ax1.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax1.legend(by_label.values(), by_label.keys(), loc="upper right")
                chart_infetti.pyplot(fig1, clear_figure=True)
                plt.close(fig1)
                
                # 2. Plot Nuovi Infetti Giornalieri
                fig2, ax2 = plt.subplots(figsize=(5, 3))
                ax2.plot(sub_df['Giorno'], sub_df['Nuovi Infetti'], color='blue', linewidth=2)
                ax2.set_title("Nuovi Contagi Giornalieri")
                ax2.set_xlim(0, max_steps)
                ax2.set_ylim(0, y_max_nuovi)
                ax2.grid(alpha=0.3)
                chart_nuovi.pyplot(fig2, clear_figure=True)
                plt.close(fig2)
                
                # 3. Plot Decessi Cumulativi
                fig3, ax3 = plt.subplots(figsize=(5, 3))
                ax3.plot(sub_df['Giorno'], sub_df['Decessi Cumulativi'], color='black', linewidth=2)
                ax3.set_title("Decessi Cumulativi")
                ax3.set_xlim(0, max_steps)
                ax3.set_ylim(0, y_max_morti)
                ax3.grid(alpha=0.3)
                chart_morti.pyplot(fig3, clear_figure=True)
                plt.close(fig3)
                
                # 4. Plot R_t
                fig4, ax4 = plt.subplots(figsize=(5, 3))
                ax4.plot(sub_df['Giorno'], sub_df['R_t'], color='purple', linewidth=2)
                ax4.axhline(y=1.0, color='green', linestyle='--', label='Soglia R_t=1')
                ax4.set_title("Indice di Trasmissibilità (R_t)")
                ax4.set_xlim(0, max_steps)
                ax4.set_ylim(0, y_max_rt)
                ax4.grid(alpha=0.3)
                ax4.legend(loc="upper right")
                chart_rt.pyplot(fig4, clear_figure=True)
                plt.close(fig4)
            
            # Mostra se al momento c'è il Lockdown
            current_lockdown_status = sub_df.iloc[-1]['Lockdown Attivo']
            if current_lockdown_status == 1:
                status_placeholder.warning(f"Giorno {i} - ⚠️ **STATO:** LOCKDOWN ATTIVO in questo momento! Le interazioni sociali sono bloccate.")
            else:
                status_placeholder.success(f"Giorno {i} - 🟢 **STATO:** CIRCOLAZIONE LIBERA all'interno della rete.")
                
            # Se la velocità è alta e i passaggi sono grandi, riduciamo il freeze visivo
            time.sleep(0.01)
    else:
        # Mostra direttamente i risultati finali
        
        # 1. Plot Infetti
        fig1, ax1 = plt.subplots(figsize=(5, 3))
        ax1.plot(df['Giorno'], df['Infetti Attuali'], color='red', label='Infetti', linewidth=2)
        lockdown_days = df[df['Lockdown Attivo'] == 1]
        if not lockdown_days.empty:
            ax1.scatter(lockdown_days['Giorno'], lockdown_days['Infetti Attuali'], color='orange', label='Lockdown', marker='s', s=10)
        ax1.set_title("Progressione Infetti Attuali")
        ax1.set_xlim(0, max_steps)
        ax1.set_ylim(0, y_max_infetti)
        ax1.grid(alpha=0.3)
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys(), loc="upper right")
        chart_infetti.pyplot(fig1, clear_figure=True)
        plt.close(fig1)
        
        # 2. Plot Nuovi Infetti
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        ax2.plot(df['Giorno'], df['Nuovi Infetti'], color='blue', linewidth=2)
        ax2.set_title("Nuovi Contagi Giornalieri")
        ax2.set_xlim(0, max_steps)
        ax2.set_ylim(0, y_max_nuovi)
        ax2.grid(alpha=0.3)
        chart_nuovi.pyplot(fig2, clear_figure=True)
        plt.close(fig2)
        
        # 3. Plot Morti
        fig3, ax3 = plt.subplots(figsize=(5, 3))
        ax3.plot(df['Giorno'], df['Decessi Cumulativi'], color='black', linewidth=2)
        ax3.set_title("Decessi Cumulativi")
        ax3.set_xlim(0, max_steps)
        ax3.set_ylim(0, y_max_morti)
        ax3.grid(alpha=0.3)
        chart_morti.pyplot(fig3, clear_figure=True)
        plt.close(fig3)
        
        # 4. Plot R_t
        fig4, ax4 = plt.subplots(figsize=(5, 3))
        ax4.plot(df['Giorno'], df['R_t'], color='purple', linewidth=2)
        ax4.axhline(y=1.0, color='green', linestyle='--', label='Soglia R_t=1')
        ax4.set_title("Indice di Trasmissibilità (R_t)")
        ax4.set_xlim(0, max_steps)
        ax4.set_ylim(0, y_max_rt)
        ax4.grid(alpha=0.3)
        ax4.legend(loc="upper right")
        chart_rt.pyplot(fig4, clear_figure=True)
        plt.close(fig4)
        
        status_placeholder.info("Simulazione completata mostrata per intero.")
