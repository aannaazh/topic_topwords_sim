import cProfile
import pstats
from sim_parallel import run_test

def profile_sim():
    # Use smaller parameters for faster profiling
    profiler = cProfile.Profile()
    profiler.enable()
    run_test(num_topics=2, alpha=0.1, beta=0.01, num_docs=100, sents_per_doc=5, iterations=3)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats(20)

if __name__ == "__main__":
    profile_sim()
