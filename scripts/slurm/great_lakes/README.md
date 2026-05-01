# Great Lakes Slurm Scripts

These scripts are specific to Jiaming Pan's Great Lakes setup.

They assume:

```text
/home/jiamingp/Diffusion_model
/home/jiamingp/Diffusion_model/cosmo_diffusion
/scratch/huterer_root/huterer0/jiamingp/saved_runs
```

For other users or clusters, copy one script and edit:

- `#SBATCH -A`
- `#SBATCH --partition`
- `#SBATCH --time`
- `cd /home/jiamingp/Diffusion_model`
- `PYTHONPATH`
- config path
- output/log paths
