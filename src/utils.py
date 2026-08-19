import time
from contextlib import contextmanager
import signal
import psutil


class PerformanceMonitor:
    """Mesure le temps d'exécution, la mémoire de départ et le pic de mémoire atteint."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.start_time = None
        self.end_time = None
        self.start_memory = None
        self.peak_memory = None
        self._process = psutil.Process()

    def start(self) -> None:
        """Démarre le monitoring."""
        self.start_time = time.time()
        self.start_memory = self._process.memory_info().rss / (1024 * 1024)  # MB
        self.peak_memory = self.start_memory

    def update_peak_memory(self) -> None:
        """À appeler pendant un traitement long pour capter un pic de mémoire intermédiaire."""
        current_memory = self._process.memory_info().rss / (1024 * 1024)
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory

    def get_metrics(self) -> dict:
        """Arrête le monitoring et retourne les métriques finales."""
        self.end_time = time.time()
        self.update_peak_memory()

        return {
            "execution_time": self.end_time - self.start_time,
            "memory_used": max(0, self.peak_memory - self.start_memory),
            "peak_memory": self.peak_memory,
            "start_memory": self.start_memory,
        }


class TimeoutException(Exception):
    pass

_timeout_frame_snapshot = []


def _snapshot_call_stack_locals(frame, max_depth=60):
    """Parcourt la pile d'appel active au moment du timeout et récupère les
    variables locales de type dict/list non vides — utile pour retrouver
    des résultats partiels accumulés par un algorithme interrompu."""
    found = []
    f = frame
    depth = 0
    while f is not None and depth < max_depth:
        for var_name, var_value in list(f.f_locals.items()):
            if isinstance(var_value, (dict, list)) and len(var_value) > 0:
                found.append((var_name, var_value, depth))
        f = f.f_back
        depth += 1
    return found


@contextmanager
def time_limit(seconds: int):
    """Context manager imposant une limite de temps d'exécution (Unix/Linux uniquement).

    En cas de dépassement, un instantané des variables locales de la pile
    d'appel est capturé avant de lever TimeoutException — récupérable via
    get_timeout_snapshot().
    """
    global _timeout_frame_snapshot
    _timeout_frame_snapshot = []

    def _handler(signum, frame):
        global _timeout_frame_snapshot
        _timeout_frame_snapshot = _snapshot_call_stack_locals(frame)
        raise TimeoutException(f"Timeout après {seconds / 3600:.1f}h")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def get_timeout_snapshot() -> list:
    """Retourne l'instantané de variables locales capturé au dernier timeout."""
    return _timeout_frame_snapshot