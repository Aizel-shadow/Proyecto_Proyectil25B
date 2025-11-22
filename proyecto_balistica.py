import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
from scipy.optimize import newton
import time

# --- LÓGICA MATEMÁTICA Y FÍSICA ---

g = 9.81

def pos_p1(t, D, h, v, phi_rad):
    """Calcula posición (x, y) del Proyectil 1."""
    x = D + v * np.cos(phi_rad) * t
    y = h + v * np.sin(phi_rad) * t - 0.5 * g * t**2
    return x, y

def calcular_parametros_intercepcion(tc, D, h, v, phi_rad, T):
    """Calcula u, theta necesarios para interceptar en tiempo tc."""
    dt = tc - T
    if dt <= 1e-3: return np.inf, 0, 0, 0
    
    target_x, target_y = pos_p1(tc, D, h, v, phi_rad)
    
    if target_y < 0: return np.inf, 0, 0, 0 # Ya chocó con el suelo
    
    ux = target_x / dt
    uy = (target_y + 0.5 * g * dt**2) / dt
    
    u = np.sqrt(ux**2 + uy**2)
    theta = np.arctan2(uy, ux)
    return u, theta, ux, uy

def derivada_energia(tc, args):
    """Derivada numérica para optimizar la energía (minima velocidad)."""
    D, h, v, phi_rad, T = args
    epsilon = 1e-5
    u1, _, _, _ = calcular_parametros_intercepcion(tc, D, h, v, phi_rad, T)
    u2, _, _, _ = calcular_parametros_intercepcion(tc + epsilon, D, h, v, phi_rad, T)
    return (u2 - u1) / epsilon

# --- MÉTODOS NUMÉRICOS ---

def metodo_newton(func, x0, args, tol=1e-6, max_iter=50):
    x = x0
    start = time.perf_counter()
    for i in range(max_iter):
        f_val = func(x, args)
        eps = 1e-5
        f_prime = (func(x + eps, args) - f_val) / eps
        if abs(f_prime) < 1e-10: break
        x_new = x - f_val / f_prime
        if abs(x_new - x) < tol:
            return x_new, i+1, (time.perf_counter() - start)*1000
        x = x_new
    return x, max_iter, (time.perf_counter() - start)*1000

def metodo_secante(func, x0, args, tol=1e-6, max_iter=50):
    x0_sec = x0 - 0.5
    x1_sec = x0 + 0.5
    start = time.perf_counter()
    for i in range(max_iter):
        f0 = func(x0_sec, args)
        f1 = func(x1_sec, args)
        if abs(f1 - f0) < 1e-10: break
        x_new = x1_sec - f1 * (x1_sec - x0_sec) / (f1 - f0)
        if abs(x_new - x1_sec) < tol:
            return x_new, i+1, (time.perf_counter() - start)*1000
        x0_sec = x1_sec
        x1_sec = x_new
    return x1_sec, max_iter, (time.perf_counter() - start)*1000

def simular_trayectorias(params, sigma_viento):
    D, h, v, phi_rad, T, u, theta_rad, tc = params
    dt = 0.02
    steps = int(tc / dt) + 10
    
    r1 = np.array([D, h], dtype=float)
    v1 = np.array([v*np.cos(phi_rad), v*np.sin(phi_rad)], dtype=float)
    traj1 = [r1.copy()]
    
    r2 = np.array([0.0, 0.0], dtype=float)
    v2 = np.array([u*np.cos(theta_rad), u*np.sin(theta_rad)], dtype=float)
    traj2 = [r2.copy()]
    
    for i in range(steps):
        t = i * dt
        noise = np.random.normal(0, sigma_viento, 2)
        
        v1[1] -= g * dt
        r1 += (v1 + noise) * dt
        traj1.append(r1.copy())
        
        if t >= T:
            v2[1] -= g * dt
            r2 += (v2 + noise) * dt
            traj2.append(r2.copy())
        else:
            traj2.append(r2.copy())
            
    return np.array(traj1), np.array(traj2)

# --- INTERFAZ GRÁFICA (GUI) ---

class AppBalistica:
    def __init__(self, root):
        self.root = root
        self.root.title("Proyecto Final: Intercepción de Proyectiles")
        self.root.geometry("1200x800")

        # Frame Principal
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Panel de Control (Izquierda) ---
        control_frame = ttk.LabelFrame(main_frame, text="Parámetros de Entrada", padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        self.crear_input(control_frame, "Distancia D (m):", "1000", "D")
        self.crear_input(control_frame, "Altura h (m):", "100", "h")
        self.crear_input(control_frame, "Velocidad P1 (m/s):", "150", "v")
        self.crear_input(control_frame, "Ángulo P1 (°):", "135", "phi")
        self.crear_input(control_frame, "Retraso T (s):", "2.0", "T")
        self.crear_input(control_frame, "Ruido Viento (σ):", "0.0", "wind")

        btn_calc = ttk.Button(control_frame, text="🚀 Calcular y Simular", command=self.ejecutar)
        btn_calc.pack(pady=20, fill=tk.X)

        # Resultados Textuales
        self.lbl_result = ttk.Label(control_frame, text="Resultados: Esperando...", wraplength=200)
        self.lbl_result.pack(pady=10)
        
        self.lbl_compare = ttk.Label(control_frame, text="", wraplength=200, font=("Arial", 8))
        self.lbl_compare.pack(pady=10)

        # --- Panel Gráfico (Derecha) ---
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.inputs = {}
        self.anim = None

    def crear_input(self, parent, label, default, key):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=5)
        ttk.Label(frame, text=label).pack(anchor="w")
        entry = ttk.Entry(frame)
        entry.insert(0, default)
        entry.pack(fill=tk.X)
        setattr(self, f"entry_{key}", entry)

    def ejecutar(self):
        try:
            # 1. Leer datos
            D = float(self.entry_D.get())
            h = float(self.entry_h.get())
            v = float(self.entry_v.get())
            phi = float(self.entry_phi.get())
            T = float(self.entry_T.get())
            wind = float(self.entry_wind.get())
            
            phi_rad = np.radians(phi)
            args = (D, h, v, phi_rad, T)
            guess = T + (D/v)*0.8 + 1.0

            # 2. Ejecutar Métodos Numéricos
            res_newton = metodo_newton(derivada_energia, guess, args)
            res_secante = metodo_secante(derivada_energia, guess, args)
            
            tc_opt, it_n, time_n = res_newton
            _, it_s, time_s = res_secante # Usamos Newton para el cálculo final

            # 3. Calcular Cinemática Final
            u_fin, theta_fin, _, _ = calcular_parametros_intercepcion(tc_opt, D, h, v, phi_rad, T)

            if np.isinf(u_fin) or np.isnan(u_fin) or tc_opt <= T:
                messagebox.showerror("Error Físico", "No es posible la intercepción con estos parámetros.\nEl objetivo está muy lejos o el retraso es muy grande.")
                return

            # 4. Actualizar UI
            self.lbl_result.config(text=f"✅ SOLUCIÓN ÓPTIMA:\n\nVelocidad P2: {u_fin:.2f} m/s\nÁngulo P2: {np.degrees(theta_fin):.2f}°\nTiempo Choque: {tc_opt:.2f} s")
            
            self.lbl_compare.config(text=f"COMPARACIÓN:\n\nNewton: {it_n} iters ({time_n:.3f} ms)\nSecante: {it_s} iters ({time_s:.3f} ms)")

            # 5. Simulación y Animación
            params_sim = (D, h, v, phi_rad, T, u_fin, theta_fin, tc_opt)
            tr1, tr2 = simular_trayectorias(params_sim, wind)
            
            self.animar(tr1, tr2)

        except ValueError:
            messagebox.showerror("Error", "Por favor ingrese solo números válidos.")

    def animar(self, tr1, tr2):
        # Limpiar gráfica anterior
        self.ax.clear()
        self.ax.set_xlabel("Distancia (m)")
        self.ax.set_ylabel("Altura (m)")
        self.ax.set_title("Simulación de Intercepción")
        self.ax.grid(True, linestyle='--', alpha=0.6)

        # Ajustar límites
        all_x = np.concatenate((tr1[:,0], tr2[:,0]))
        all_y = np.concatenate((tr1[:,1], tr2[:,1]))
        self.ax.set_xlim(min(0, np.min(all_x))-50, np.max(all_x)+50)
        self.ax.set_ylim(0, max(10, np.max(all_y)+50))

        # Elementos gráficos
        line1, = self.ax.plot([], [], 'b--', label='Objetivo')
        pt1, = self.ax.plot([], [], 'bo')
        line2, = self.ax.plot([], [], 'r-', label='Interceptor')
        pt2, = self.ax.plot([], [], 'ro')
        self.ax.legend()

        # Lógica de animación
        step = max(1, len(tr1) // 100) # Optimización de frames

        def update(frame):
            idx = frame * step
            i1 = min(idx, len(tr1)-1)
            i2 = min(idx, len(tr2)-1)

            line1.set_data(tr1[:i1, 0], tr1[:i1, 1])
            pt1.set_data([tr1[i1, 0]], [tr1[i1, 1]])
            line2.set_data(tr2[:i2, 0], tr2[:i2, 1])
            pt2.set_data([tr2[i2, 0]], [tr2[i2, 1]])
            return line1, pt1, line2, pt2

        # Importante: Guardar referencia a self.anim
        if self.anim: self.anim.event_source.stop()
        
        frames = max(len(tr1), len(tr2)) // step
        self.anim = FuncAnimation(self.fig, update, frames=frames, interval=20, blit=False)
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = AppBalistica(root)
    root.mainloop()