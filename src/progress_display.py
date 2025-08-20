import time
import sys
from datetime import datetime

# Try to import colorama, fallback to no colors if not available
try:
    from colorama import Fore, Back, Style, init
    init(autoreset=True)
    HAS_COLOR = True
except ImportError:
    # Fallback - no colors
    class _NoColor:
        def __getattr__(self, name):
            return ""
    
    Fore = _NoColor()
    Back = _NoColor()  
    Style = _NoColor()
    HAS_COLOR = False

class HackerProgressDisplay:
    """
    Fancy hacker-style progress display for portfolio optimization
    """
    
    def __init__(self):
        self.start_time = None
        self.current_phase = None
        self.total_steps = 0
        self.current_step = 0
        
    def print_header(self):
        """Print cool ASCII header"""
        header = f"""
{Fore.GREEN}{Style.BRIGHT}
╔══════════════════════════════════════════════════════════════════════════════╗
║  ▄████▄  ▄▄▄█████▓  ▄████  ▄▄▄      ███▄    █    ██▓███   ▄▄▄       █     ║
║ ▒██▀ ▀█  ▓  ██▒ ▓▒ ██▒ ▀█▒▒████▄    ██ ▀█   █   ▓██░  ██▒▒████▄     █     ║
║ ▒▓█    ▄ ▒ ▓██░ ▒░▒██░▄▄▄░▒██  ▀█▄ ▓██  ▀█ ██▒  ▓██░ ██▓▒▒██  ▀█▄   █     ║
║ ▒▓▓▄ ▄██▒░ ▓██▓ ░ ░▓█  ██▓░██▄▄▄▄██▓██▒  ▐▌██▒  ▒██▄█▓▒ ▒░██▄▄▄▄██  █     ║
║ ▒ ▓███▀ ░  ▒██▒ ░ ░▒▓███▀▒ ▓█   ▓██▒██░   ▓██░  ▒██▒ ░  ░ ▓█   ▓██▒ █     ║
║ ░ ░▒ ▒  ░  ▒ ░░    ░▒   ▒  ▒▒   ▓▒█░ ▒░   ▒ ▒   ▒▓▒░ ░  ░ ▒▒   ▓▒█░ ░     ║
║   ░  ▒       ░      ░   ░   ▒   ▒▒ ░ ░░   ░ ▒░  ░▒ ░       ▒   ▒▒ ░ ░     ║
║ ░          ░      ░ ░   ░   ░   ▒     ░   ░ ░   ░░         ░   ▒    ░     ║
║ ░ ░               ░         ░  ░        ░                   ░  ░          ║
║                                                                            ║
║          {Fore.CYAN}PORTFOLIO OPTIMIZATION ENGINE v2.0{Fore.GREEN}                          ║
║                    {Fore.YELLOW}CTGAN + HISTORICAL SAMPLING{Fore.GREEN}                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
{Style.RESET_ALL}
"""
        print(header)
        
    def start_phase(self, phase_name, total_steps):
        """Start a new processing phase"""
        self.current_phase = phase_name
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        
        print(f"\n{Fore.CYAN}{'='*80}")
        print(f"{Fore.GREEN}[{datetime.now().strftime('%H:%M:%S')}] {Fore.YELLOW}INITIATING: {phase_name}")
        print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
        
    def update_progress(self, current_date=None, model_name=None, sub_task=None):
        """Update progress with current processing info"""
        self.current_step += 1
        
        # Calculate progress
        progress = self.current_step / self.total_steps if self.total_steps > 0 else 0
        elapsed = time.time() - self.start_time if self.start_time else 0
        eta = (elapsed / progress - elapsed) if progress > 0 else 0
        
        # Create progress bar
        bar_length = 50
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        # Format time
        eta_str = f"{int(eta//60):02d}:{int(eta%60):02d}" if eta > 0 else "∞"
        elapsed_str = f"{int(elapsed//60):02d}:{int(elapsed%60):02d}"
        
        # Build status line
        status_parts = []
        if current_date:
            status_parts.append(f"{Fore.YELLOW}📅 {current_date.strftime('%Y-%m-%d')}")
        if model_name:
            status_parts.append(f"{Fore.MAGENTA}🔬 {model_name}")
        if sub_task:
            status_parts.append(f"{Fore.CYAN}⚙️ {sub_task}")
            
        status = " │ ".join(status_parts) if status_parts else "Processing..."
        
        # Clear line and print progress
        print(f"\r{' ' * 120}", end='')  # Clear line
        print(f"\r{Fore.GREEN}[{progress:6.1%}] {Fore.WHITE}{bar} {Fore.GREEN}│ "
              f"{Fore.BLUE}T: {elapsed_str} │ ETA: {eta_str} │ {status}", end='', flush=True)
              
    def complete_phase(self):
        """Complete current phase"""
        elapsed = time.time() - self.start_time if self.start_time else 0
        elapsed_str = f"{int(elapsed//60):02d}:{int(elapsed%60):02d}"
        
        print(f"\n{Fore.GREEN}✅ PHASE COMPLETE: {self.current_phase}")
        print(f"{Fore.BLUE}⏱️  Total Time: {elapsed_str}")
        print(f"{Fore.CYAN}{'─'*80}{Style.RESET_ALL}\n")
        
    def print_results_header(self):
        """Print results section header"""
        print(f"""
{Fore.GREEN}╔══════════════════════════════════════════════════════════════════════════════╗
║                            {Fore.YELLOW}OPTIMIZATION RESULTS{Fore.GREEN}                             ║
║                         {Fore.CYAN}Portfolio Performance Analysis{Fore.GREEN}                        ║
╚══════════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
""")

    def print_model_results(self, model_name, metrics):
        """Print formatted model results"""
        print(f"\n{Fore.MAGENTA}● MODEL: {model_name.upper()}")
        print(f"{Fore.CYAN}├─ 📈 Annualized Return: {Fore.GREEN}{metrics.get('return', 'N/A')}")
        print(f"{Fore.CYAN}├─ ⚠️  CVaR (Ex-post): {Fore.YELLOW}{metrics.get('cvar', 'N/A')}")
        print(f"{Fore.CYAN}├─ 🎯 Mean HHI: {Fore.BLUE}{metrics.get('hhi', 'N/A')}")
        print(f"{Fore.CYAN}└─ 🔄 Mean Rotation: {Fore.WHITE}{metrics.get('rotation', 'N/A')}")
        
    def print_visualization_status(self, save_path):
        """Print visualization completion status"""
        print(f"""
{Fore.GREEN}╔══════════════════════════════════════════════════════════════════════════════╗
║                          {Fore.YELLOW}VISUALIZATION COMPLETE{Fore.GREEN}                            ║
║                     {Fore.CYAN}📊 Charts Generated Successfully{Fore.GREEN}                         ║
║                                                                            ║
║  {Fore.WHITE}📁 Location: {save_path:<58} {Fore.GREEN}║
╚══════════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
""")

    def print_footer(self):
        """Print completion footer"""
        print(f"""
{Fore.GREEN}
╔══════════════════════════════════════════════════════════════════════════════╗
║                            {Fore.YELLOW}MISSION ACCOMPLISHED{Fore.GREEN}                             ║
║                    {Fore.CYAN}Portfolio optimization sequence complete{Fore.GREEN}                  ║
║                                                                            ║
║           {Fore.WHITE}🎯 CVaR constraints enforced  📊 Visualizations ready{Fore.GREEN}            ║
║           {Fore.WHITE}⚡ Performance optimized      🔬 Models compared{Fore.GREEN}                 ║
╚══════════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}

{Fore.GREEN}[{datetime.now().strftime('%H:%M:%S')}] {Fore.CYAN}>>> {Fore.WHITE}System ready for next operation{Style.RESET_ALL}
""")