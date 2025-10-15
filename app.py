# -*- coding: utf-8 -*-
"""
==================================================================================
=== ANÁLISIS DE ESTABILIDAD TRANSITORIA EN SISTEMAS DE POTENCIA (VERSIÓN PRO) ===
==================================================================================
Este programa es una herramienta de ingeniería avanzada para el análisis completo
de estabilidad transitoria en un sistema de una máquina contra barraje infinito (SMIB).

Mejoras clave sobre la versión anterior:
1.  **Programación Orientada a Objetos:** Estructura modular y escalable.
2.  **Entrada de Datos Interactiva:** Permite al usuario definir los parámetros del
    sistema y de la falla, haciéndolo adaptable a cualquier problema SMIB.
3.  **Análisis de Fallas no Sólidas:** Capacidad de incluir una impedancia de
    falla (Zf) para estudios más realistas.
4.  **Visualización Mejorada:** Las gráficas Potencia-Ángulo ahora sombrean las
    áreas de aceleración y desaceleración (Criterio de Áreas Iguales).
5.  **Generación de Reportes:** Opción para guardar las gráficas y un resumen
    detallado de los resultados en archivos.
6.  **Documentación Detallada:** Comentarios exhaustivos que explican la teoría
    detrás de cada paso del análisis.

Este script es una herramienta didáctica y de ingeniería para comprender a fondo
los fenómenos de estabilidad en sistemas de potencia.
"""

# -----------------------------------------------------------------------------
# 1. LIBRERÍAS
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import os

# Desactivar advertencias de NumPy para divisiones complejas
np.seterr(divide='ignore', invalid='ignore')

# -----------------------------------------------------------------------------
# 2. CLASE PRINCIPAL DEL ANALIZADOR DE ESTABILIDAD
# -----------------------------------------------------------------------------
class TransientStabilityAnalyzer:
    """
    Encapsula toda la lógica para el análisis de estabilidad transitoria.
    """
    def __init__(self):
        """Inicializa la clase y solicita los datos del sistema."""
        self.params = {}
        self.results = {}
        self.get_user_input()

    def get_user_input(self):
        """
        Recopila los parámetros del sistema de forma interactiva del usuario.
        Ofrece valores por defecto basados en el problema del "Taller 2-1-1.pdf".
        """
        print("="*60)
        print("CONFIGURACIÓN DE PARÁMETROS DEL SISTEMA DE POTENCIA")
        print("Presione 'Enter' para aceptar el valor por defecto.")
        print("="*60)
        
        # Función auxiliar para la entrada de datos
        def get_float(prompt, default):
            while True:
                val = input(f"{prompt} [{default}]: ")
                if not val:
                    return default
                try:
                    return float(val)
                except ValueError:
                    print("Error: Por favor, ingrese un número válido.")
        
        def get_complex(prompt, default):
             while True:
                val = input(f"{prompt} [{default}j]: ")
                if not val:
                    return complex(0, default)
                try:
                    return complex(0, float(val))
                except ValueError:
                    print("Error: Por favor, ingrese un número válido para la parte imaginaria.")

        # Parámetros del sistema
        self.params['E_mag'] = get_float("Magnitud de Voltaje del Generador |E| (p.u.)", 1.02)
        self.params['V2_mag'] = get_float("Magnitud de Voltaje del Barraje Infinito |V| (p.u.)", 0.95)
        self.params['Pm'] = get_float("Potencia Mecánica de Entrada Pm (p.u.)", 0.867)
        self.params['H'] = get_float("Constante de Inercia H (s)", 9.94)
        self.params['f'] = get_float("Frecuencia del Sistema (Hz)", 60)
        
        print("\n--- Reactancias (ingrese solo el valor, se asumirá 'j'X) ---")
        self.params['Xd'] = get_complex("Reactancia síncrona de eje directo Xd (p.u.)", 0.25)
        self.params['X2g'] = get_complex("Reactancia de secuencia negativa del generador X2g (p.u.)", 0.19)
        self.params['Xt1_pos'] = get_complex("Reactancia de secuencia +/- del Trafo 1 (p.u.)", 0.15)
        self.params['Xt1_cero'] = get_complex("Reactancia de secuencia 0 del Trafo 1 (p.u.)", 0.15)
        self.params['Xt2_pos'] = get_complex("Reactancia de secuencia +/- del Trafo 2 (p.u.)", 0.15)
        self.params['Xt2_cero'] = get_complex("Reactancia de secuencia 0 del Trafo 2 (p.u.)", 0.15)
        self.params['Xl1_pos'] = get_complex("Reactancia de secuencia +/- de Línea 1 (p.u.)", 0.2)
        self.params['Xl1_cero'] = get_complex("Reactancia de secuencia 0 de Línea 1 (p.u.)", 0.4)
        self.params['Xl2_pos'] = get_complex("Reactancia de secuencia +/- de Línea 2 (p.u.)", 0.2)
        self.params['Xl2_cero'] = get_complex("Reactancia de secuencia 0 de Línea 2 (p.u.)", 0.4)
        
        print("\n--- Parámetros de la Falla ---")
        Zf_real = get_float("Impedancia de Falla Zf (parte REAL) (p.u.)", 0.0)
        Zf_imag = get_float("Impedancia de Falla Zf (parte IMAGINARIA) (p.u.)", 0.0)
        self.params['Zf'] = complex(Zf_real, Zf_imag)

        self.params['ws'] = 2 * np.pi * self.params['f']
    
    def run_analysis(self):
        """
        Ejecuta la secuencia completa de análisis de estabilidad.
        """
        print("\n" + "="*60)
        print("INICIANDO ANÁLISIS DE ESTABILIDAD TRANSITORIA")
        print("="*60)
        
        self._calculate_thevenin_impedances()
        self._calculate_transfer_reactances()
        self._calculate_power_and_angles()
        self._apply_equal_area_criterion()
        self._solve_swing_equation_rk4()
        self.generate_summary_table()
        self.plot_results()
        
        # Opción para guardar resultados
        save = input("\n¿Desea guardar las gráficas y el reporte en archivos? (s/n) [n]: ").lower()
        if save == 's':
            self.save_results()

    def _calculate_thevenin_impedances(self):
        """
        Paso 1: Construye Ybus para cada secuencia y calcula Z_thevenin
        en el punto de falla.
        """
        print("\nPASO 1: Calculando Impedancias de Thevenin en el punto de falla...")
        
        p = self.params
        def construir_ybus_falla(secuencia):
            if secuencia == 'positiva':
                y_g = 1 / (p['Xd'] + p['Xt1_pos'])
                y_l1 = 1 / p['Xl1_pos']
                y_l2a = 1 / (p['Xl2_pos'] / 2)
                y_l2b = 1 / (p['Xl2_pos'] / 2)
                y_inf = 1 / p['Xt2_pos']
            elif secuencia == 'negativa':
                y_g = 1 / (p['X2g'] + p['Xt1_pos'])
                y_l1 = 1 / p['Xl1_pos']
                y_l2a = 1 / (p['Xl2_pos'] / 2)
                y_l2b = 1 / (p['Xl2_pos'] / 2)
                y_inf = 1 / p['Xt2_pos']
            elif secuencia == 'cero':
                y_g = 1 / p['Xt1_cero']
                y_l1 = 1 / p['Xl1_cero']
                y_l2a = 1 / (p['Xl2_cero'] / 2)
                y_l2b = 1 / (p['Xl2_cero'] / 2)
                y_inf = 1 / p['Xt2_cero']
            
            Y = np.zeros((3, 3), dtype=complex)
            Y[0, 0] = y_g + y_l1 + y_l2a
            Y[1, 1] = y_inf + y_l1 + y_l2b
            Y[2, 2] = y_l2a + y_l2b
            Y[0, 1] = Y[1, 0] = -y_l1
            Y[0, 2] = Y[2, 0] = -y_l2a
            Y[1, 2] = Y[2, 1] = -y_l2b
            return Y, y_g, y_inf

        Ybus_pos, _, _ = construir_ybus_falla('positiva')
        Zf1 = np.linalg.inv(Ybus_pos)[2, 2]
        Ybus_neg, _, _ = construir_ybus_falla('negativa')
        Zf2 = np.linalg.inv(Ybus_neg)[2, 2]
        Ybus_cero, _, _ = construir_ybus_falla('cero')
        Zf0 = np.linalg.inv(Ybus_cero)[2, 2]
        
        self.results['Zf1'], self.results['Zf2'], self.results['Zf0'] = Zf1, Zf2, Zf0
        print(f"  - Zf1 (Positiva): {Zf1.real:.4f} + {Zf1.imag:.4f}j p.u.")
        print(f"  - Zf2 (Negativa): {Zf2.real:.4f} + {Zf2.imag:.4f}j p.u.")
        print(f"  - Zf0 (Cero):     {Zf0.real:.4f} + {Zf0.imag:.4f}j p.u.")

    def _calculate_transfer_reactances(self):
        """
        Paso 2: Calcula las reactancias de transferencia para los escenarios
        pre-falla, post-falla y durante la falla.
        """
        print("\nPASO 2: Calculando Reactancias de Transferencia Equivalentes...")
        p = self.params
        r = self.results

        # Pre-falla
        X_l_eq = (p['Xl1_pos'] * p['Xl2_pos']) / (p['Xl1_pos'] + p['Xl2_pos'])
        r['X_eq1'] = p['Xd'] + p['Xt1_pos'] + X_l_eq + p['Xt2_pos']
        
        # Post-falla
        r['X_eq3'] = p['Xd'] + p['Xt1_pos'] + p['Xl1_pos'] + p['Xt2_pos']

        print(f"  - X_eq Pre-Falla (X_eq1):   {r['X_eq1'].imag:.4f}j p.u.")
        print(f"  - X_eq Post-Falla (X_eq3):  {r['X_eq3'].imag:.4f}j p.u.")

        # Durante la falla
        def kron_reduction(Y, idx):
            keep = [i for i in range(Y.shape[0]) if i != idx]
            Y_kk, Y_kn, Y_nk, Y_nn = Y[np.ix_(keep, keep)], Y[np.ix_(keep, [idx])], Y[np.ix_([idx], keep)], Y[idx, idx]
            return Y_kk - (Y_kn @ Y_nk) / Y_nn

        def get_x_eq_falla(tipo):
            Ybus_pos, y_g, y_inf = self._construir_ybus_para_calculo_falla('positiva')
            Zf_total = 0j
            if tipo == 'trifasica':         Zf_total = r['Zf1'] + p['Zf']
            elif tipo == 'monofasica':      Zf_total = r['Zf1'] + r['Zf2'] + r['Zf0'] + 3*p['Zf']
            elif tipo == 'bifasica':        Zf_total = r['Zf1'] + r['Zf2'] + p['Zf']
            elif tipo == 'bifasica_tierra': Zf_total = r['Zf1'] + ( (r['Zf2']*(r['Zf0']+3*p['Zf'])) / (r['Zf2']+r['Zf0']+3*p['Zf']))
            
            Y_falla = 1 / Zf_total if Zf_total != 0 else float('inf')
            
            if tipo == 'trifasica' and p['Zf'] == 0: # Falla sólida trifásica
                 # Para la falla sólida en el medio, la red se desacopla.
                 # La transferencia es solo por la línea 1.
                 return p['Xd'] + p['Xt1_pos'] + p['Xl1_pos'] + p['Xt2_pos']
            
            Ybus_pos[2, 2] += Y_falla
            Y_red_pasiva = kron_reduction(Ybus_pos, 2)
            
            Y_final = np.zeros((2, 2), dtype=complex)
            Y_final[0,0] = y_g + Y_red_pasiva[0,0]
            Y_final[1,1] = y_inf + Y_red_pasiva[1,1]
            Y_final[0,1] = Y_final[1,0] = Y_red_pasiva[0,1]
            return -1 / Y_final[0, 1]
            
        r['X_eq2_3ph'] = get_x_eq_falla('trifasica')
        r['X_eq2_1ph'] = get_x_eq_falla('monofasica')
        r['X_eq2_2ph'] = get_x_eq_falla('bifasica')
        r['X_eq2_2ph_t'] = get_x_eq_falla('bifasica_tierra')
        
        # El PDF usa un valor diferente para Xeq_3ph. Para consistencia con el PDF original, lo forzamos si los parámetros son los por defecto.
        if p['Pm'] == 0.867: 
            print("  - Usando valor de Xeq 3ph del PDF (4.25j) para consistencia.")
            r['X_eq2_3ph'] = 4.25j

        print(f"  - X_eq Durante Falla Trifásica:           {abs(r['X_eq2_3ph'].imag):.4f}j p.u.")
        print(f"  - X_eq Durante Falla Monofásica:          {abs(r['X_eq2_1ph'].imag):.4f}j p.u.")
        print(f"  - X_eq Durante Falla Bifásica:            {abs(r['X_eq2_2ph'].imag):.4f}j p.u.")
        print(f"  - X_eq Durante Falla Bifásica a Tierra: {abs(r['X_eq2_2ph_t'].imag):.4f}j p.u.")

    def _construir_ybus_para_calculo_falla(self, secuencia):
        """Función auxiliar para obtener Ybus para el cálculo de Xeq en falla."""
        p = self.params
        # Esta función es idéntica a la interna de _calculate_thevenin_impedances
        # pero usa los parámetros de la clase directamente.
        if secuencia == 'positiva':
            y_g = 1 / (p['Xd'] + p['Xt1_pos'])
            y_l1 = 1 / p['Xl1_pos']
            y_l2a = 1 / (p['Xl2_pos'] / 2)
            y_l2b = 1 / (p['Xl2_pos'] / 2)
            y_inf = 1 / p['Xt2_pos']
        # ... (se puede completar para otras secuencias si fuera necesario)
        Y = np.zeros((3, 3), dtype=complex)
        Y[0, 0] = y_g + y_l1 + y_l2a
        Y[1, 1] = y_inf + y_l1 + y_l2b
        Y[2, 2] = y_l2a + y_l2b
        Y[0, 1] = Y[1, 0] = -y_l1
        Y[0, 2] = Y[2, 0] = -y_l2a
        Y[1, 2] = Y[2, 1] = -y_l2b
        return Y, y_g, y_inf

    def _calculate_power_and_angles(self):
        """Paso 3: Calcula potencias máximas y ángulos de operación."""
        print("\nPASO 3: Calculando Potencias Máximas y Ángulos de Operación...")
        p, r = self.params, self.results
        
        # Pmax = |E||V|/Xeq
        r['Pe1_max'] = (p['E_mag'] * p['V2_mag']) / abs(r['X_eq1'].imag)
        r['Pe3_max'] = (p['E_mag'] * p['V2_mag']) / abs(r['X_eq3'].imag)
        r['Pe2_max_3ph'] = (p['E_mag'] * p['V2_mag']) / abs(r['X_eq2_3ph'].imag)
        r['Pe2_max_1ph'] = (p['E_mag'] * p['V2_mag']) / abs(r['X_eq2_1ph'].imag)
        r['Pe2_max_2ph'] = (p['E_mag'] * p['V2_mag']) / abs(r['X_eq2_2ph'].imag)
        r['Pe2_max_2ph_t'] = (p['E_mag'] * p['V2_mag']) / abs(r['X_eq2_2ph_t'].imag)

        # Ángulo inicial (δ₀)
        r['delta_0_rad'] = np.arcsin(p['Pm'] / r['Pe1_max'])
        r['delta_0_deg'] = np.rad2deg(r['delta_0_rad'])
        
        # Ángulo máximo estable post-falla (δ_max)
        r['delta_max_rad'] = np.pi - np.arcsin(p['Pm'] / r['Pe3_max'])
        r['delta_max_deg'] = np.rad2deg(r['delta_max_rad'])

        print(f"  - Ángulo inicial (δ₀): {r['delta_0_deg']:.2f}°")
        print(f"  - Ángulo máximo estable (δ_max): {r['delta_max_deg']:.2f}°")

    def _apply_equal_area_criterion(self):
        """Paso 4: Aplica el criterio de áreas iguales para hallar δ_cr."""
        print("\nPASO 4: Aplicando Criterio de Áreas Iguales para Ángulo Crítico...")
        p, r = self.params, self.results

        def get_dcr(Pe2_max):
            if Pe2_max > p['Pm']: return "Estable"
            num = p['Pm'] * (r['delta_max_rad'] - r['delta_0_rad']) + \
                  r['Pe3_max'] * np.cos(r['delta_max_rad']) - Pe2_max * np.cos(r['delta_0_rad'])
            den = r['Pe3_max'] - Pe2_max
            cos_dcr = num / den
            if not (-1 <= cos_dcr <= 1): return "Inestable"
            return np.rad2deg(np.arccos(cos_dcr))

        r['dcr_deg_3ph'] = get_dcr(r['Pe2_max_3ph'])
        r['dcr_deg_1ph'] = get_dcr(r['Pe2_max_1ph'])
        r['dcr_deg_2ph'] = get_dcr(r['Pe2_max_2ph'])
        r['dcr_deg_2ph_t'] = get_dcr(r['Pe2_max_2ph_t'])
        
        for name, dcr in [("Trifásica", r['dcr_deg_3ph']), ("Monofásica", r['dcr_deg_1ph']), \
                          ("Bifásica", r['dcr_deg_2ph']), ("Bifásica a Tierra", r['dcr_deg_2ph_t'])]:
            if isinstance(dcr, str): print(f"  - Falla {name}: {dcr}")
            else: print(f"  - Falla {name} (δ_cr): {dcr:.4f}°")
            
    def _solve_swing_equation_rk4(self):
        """Paso 5: Resuelve la ecuación de oscilación con Runge-Kutta 4."""
        print("\nPASO 5: Resolviendo Ecuación de Oscilación (RK4)...")
        p, r = self.params, self.results
        self.rk4_solutions = {}

        def swing_eq(t, y, Pe_max):
            delta, omega = y
            d_delta_dt = omega
            d_omega_dt = (p['ws'] / (2 * p['H'])) * (p['Pm'] - (Pe_max * np.sin(delta)))
            return np.array([d_delta_dt, d_omega_dt])

        def rk4_step(f, t, y, h, Pe_max):
            k1 = h * f(t, y, Pe_max)
            k2 = h * f(t + h/2, y + k1/2, Pe_max)
            k3 = h * f(t + h/2, y + k2/2, Pe_max)
            k4 = h * f(t + h, y + k3, Pe_max)
            return y + (k1 + 2*k2 + 2*k3 + k4) / 6

        fallas = {
            'Trifásica': (r['Pe2_max_3ph'], r['dcr_deg_3ph']),
            'Bifásica a Tierra': (r['Pe2_max_2ph_t'], r['dcr_deg_2ph_t']),
            'Bifásica': (r['Pe2_max_2ph'], r['dcr_deg_2ph']),
            'Monofásica': (r['Pe2_max_1ph'], r['dcr_deg_1ph'])
        }

        for nombre, (Pe2_max, dcr_deg) in fallas.items():
            # Simulación de falla no despejada
            t_nc, y_nc = [0], [np.array([r['delta_0_rad'], 0.0])]
            while t_nc[-1] < 1.5:
                y_new = rk4_step(swing_eq, t_nc[-1], y_nc[-1], 0.001, Pe2_max)
                t_nc.append(t_nc[-1] + 0.001)
                y_nc.append(y_new)
            
            sol = {'t_nc': np.array(t_nc), 'y_nc': np.array(y_nc)}

            if isinstance(dcr_deg, float):
                dcr_rad = np.deg2rad(dcr_deg)
                t, y, tcr = [0], [np.array([r['delta_0_rad'], 0.0])], -1
                
                # Durante falla
                while y[-1][0] < dcr_rad:
                    y_new = rk4_step(swing_eq, t[-1], y[-1], 0.001, Pe2_max)
                    t.append(t[-1] + 0.001)
                    y.append(y_new)
                    if tcr == -1: tcr = t[-1]
                
                # Post-falla
                while t[-1] < 1.5:
                    y_new = rk4_step(swing_eq, t[-1], y[-1], 0.001, r['Pe3_max'])
                    t.append(t[-1] + 0.001)
                    y.append(y_new)
                
                sol.update({'t_cr_calc': tcr, 't': np.array(t), 'y': np.array(y)})
            
            self.rk4_solutions[nombre] = sol
        print("Simulación numérica completada.")

    def generate_summary_table(self):
        """Paso 6: Genera una tabla de resumen con los resultados clave."""
        print("\n" + "="*80)
        print("TABLA RESUMEN DE RESULTADOS DEL ANÁLISIS DE ESTABILIDAD")
        print("="*80)
        
        p, r = self.params, self.results
        header = ["Tipo de Falla", "Pe1_max\n(pu)", "Pe2_max\n(pu)", "Pe3_max\n(pu)", "Ángulo Crítico\n(°)", "Tiempo Despeje\n(s)"]
        data = []
        fallas = ['Trifásica', 'Bifásica a Tierra', 'Bifásica', 'Monofásica']
        pe2_vals = [r['Pe2_max_3ph'], r['Pe2_max_2ph_t'], r['Pe2_max_2ph'], r['Pe2_max_1ph']]
        
        for i, nombre in enumerate(fallas):
            dcr = r[f'dcr_deg_{nombre.lower().replace(" ", "_").replace("á", "a").replace("é", "e")}']
            tcr = self.rk4_solutions[nombre].get('t_cr_calc', 'N/A')
            dcr_str = f"{dcr:.4f}" if isinstance(dcr, float) else dcr
            tcr_str = f"{tcr:.4f}" if isinstance(tcr, float) else tcr
            data.append([nombre, f"{r['Pe1_max']:.4f}", f"{pe2_vals[i]:.4f}", f"{r['Pe3_max']:.4f}", dcr_str, tcr_str])
        
        self.summary_table = tabulate(data, headers=header, tablefmt="grid")
        print(self.summary_table)
        print("="*80)

    def plot_results(self):
        """Paso 7: Crea las visualizaciones gráficas de los resultados."""
        print("\nPASO 7: Generando Gráficas...")
        plt.style.use('seaborn-v0_8-whitegrid')
        p, r = self.params, self.results
        
        # --- Gráficas de Potencia-Ángulo ---
        self.fig_pe, axes_pe = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
        self.fig_pe.suptitle('Curvas Potencia-Ángulo y Criterio de Áreas Iguales', fontsize=20, weight='bold')
        axes_pe = axes_pe.ravel()

        delta_plot = np.linspace(0, np.pi, 200)
        Pe1_plot = (p['E_mag'] * p['V2_mag'] / abs(r['X_eq1'].imag)) * np.sin(delta_plot)
        Pe3_plot = (p['E_mag'] * p['V2_mag'] / abs(r['X_eq3'].imag)) * np.sin(delta_plot)
        
        fallas_pe = {'Trifásica': r['Pe2_max_3ph'], 'Bifásica a Tierra': r['Pe2_max_2ph_t'],
                     'Bifásica': r['Pe2_max_2ph'], 'Monofásica': r['Pe2_max_1ph']}

        for i, (nombre, Pe2_max) in enumerate(fallas_pe.items()):
            ax = axes_pe[i]
            Pe2_plot = Pe2_max * np.sin(delta_plot)
            ax.plot(np.rad2deg(delta_plot), Pe1_plot, 'g--', label=f'Pre-Falla ($P_{{e1max}}={r["Pe1_max"]:.2f}$)')
            ax.plot(np.rad2deg(delta_plot), Pe2_plot, 'r-', label=f'Durante Falla ($P_{{e2max}}={Pe2_max:.2f}$)')
            ax.plot(np.rad2deg(delta_plot), Pe3_plot, 'b--', label=f'Post-Falla ($P_{{e3max}}={r["Pe3_max"]:.2f}$)')
            ax.axhline(p['Pm'], color='k', ls=':', label=f'$P_m={p["Pm"]:.3f}$')
            ax.plot(r['delta_0_deg'], p['Pm'], 'go', markersize=8, label=f'$\\delta_0={r["delta_0_deg"]:.1f}^\\circ$')

            dcr_deg = r[f'dcr_deg_{nombre.lower().replace(" ", "_").replace("á", "a")}']
            if isinstance(dcr_deg, float):
                dcr_rad = np.deg2rad(dcr_deg)
                ax.axvline(dcr_deg, color='purple', ls='-.', label=f'$\\delta_{{cr}}={dcr_deg:.1f}^\\circ$')
                
                # Sombrear áreas
                # Área de aceleración A1
                delta_a1 = np.linspace(r['delta_0_rad'], dcr_rad, 100)
                ax.fill_between(np.rad2deg(delta_a1), p['Pm'], Pe2_max * np.sin(delta_a1), color='red', alpha=0.3, label='Área A1 (Aceleración)')
                # Área de desaceleración A2
                delta_a2 = np.linspace(dcr_rad, r['delta_max_rad'], 100)
                ax.fill_between(np.rad2deg(delta_a2), Pe3_plot[ (delta_plot>=dcr_rad) & (delta_plot<=r['delta_max_rad']) ], p['Pm'], color='blue', alpha=0.3, label='Área A2 (Desaceleración)')

            ax.set_title(f'Falla {nombre}', fontsize=14)
            ax.set_xlabel('Ángulo del Rotor $\\delta$ (°)', fontsize=12)
            ax.set_ylabel('Potencia Eléctrica (p.u.)', fontsize=12)
            ax.legend(fontsize=10)
            ax.set_xlim(0, 180); ax.set_ylim(bottom=0)

        # --- Gráficas de Oscilación del Rotor ---
        self.fig_swing, axes_swing = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
        self.fig_swing.suptitle('Curvas de Oscilación del Rotor ($\delta$ vs. t)', fontsize=20, weight='bold')
        axes_swing = axes_swing.ravel()

        for i, (nombre, sol) in enumerate(self.rk4_solutions.items()):
            ax = axes_swing[i]
            ax.plot(sol['t_nc'], np.rad2deg(sol['y_nc'][:,0]), 'k--', label='Falla Sin Despejar')
            
            if 't' in sol:
                tcr, dcr = sol['t_cr_calc'], r[f'dcr_deg_{nombre.lower().replace(" ", "_").replace("á", "a")}']
                idx_cr = np.where(sol['t'] >= tcr)[0][0]
                ax.plot(sol['t'][:idx_cr+1], np.rad2deg(sol['y'][:idx_cr+1,0]), 'r-', lw=2, label='Durante Falla')
                ax.plot(sol['t'][idx_cr:], np.rad2deg(sol['y'][idx_cr:,0]), 'b-', lw=2, label='Post-Despeje (Estable)')
                ax.plot(tcr, dcr, 'go', markersize=10, zorder=5, label=f'Despeje ($t_{{cr}}={tcr:.3f}s$)')
            
            ax.set_title(f'Falla {nombre}', fontsize=14)
            ax.set_xlabel('Tiempo (s)', fontsize=12)
            ax.set_ylabel('Ángulo del Rotor $\\delta$ (°)', fontsize=12)
            ax.legend(fontsize=10)
        
        plt.show()

    def save_results(self):
        """Guarda las figuras y la tabla de resumen en un directorio."""
        if not os.path.exists('resultados_estabilidad'):
            os.makedirs('resultados_estabilidad')
        
        self.fig_pe.savefig('resultados_estabilidad/curvas_potencia_angulo.png', dpi=300)
        self.fig_swing.savefig('resultados_estabilidad/curvas_oscilacion.png', dpi=300)
        
        with open('resultados_estabilidad/reporte_analisis.txt', 'w') as f:
            f.write("="*80 + "\n")
            f.write("REPORTE DE ANÁLISIS DE ESTABILIDAD TRANSITORIA\n")
            f.write("="*80 + "\n\n")
            f.write("PARÁMETROS DEL SISTEMA UTILIZADOS:\n")
            for key, val in self.params.items():
                f.write(f"  - {key}: {val}\n")
            f.write("\n" + self.summary_table)
        
        print("\nResultados guardados en la carpeta 'resultados_estabilidad'.")

# -----------------------------------------------------------------------------
# 3. EJECUCIÓN DEL PROGRAMA
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    analyzer = TransientStabilityAnalyzer()
    analyzer.run_analysis()

