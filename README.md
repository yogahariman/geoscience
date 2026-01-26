# Hubungkan local → GitHub
cd /Drive/D/geoscience

## sekali saja untuk init
git init
git status
git add .
git commit -m "Initial commit: geoscience toolbox"
git branch -M main
git remote add origin https://github.com/yogahariman/geoscience.git
git push -u origin main

# cara pakai
git clone https://github.com/username/geoscience.git
cd geoscience
pip install -e .

geoscience/                     ← repo name & pip package name
├── geosc/                      ← python namespace
│   ├── __init__.py
│   │
│   ├── seismic/
│   │   ├── __init__.py
│   │   │
│   │   ├── segy/               ← SEG-Y I/O & low-level tools
│   │   │   ├── __init__.py
│   │   │   ├── reader.py       ← read SEG-Y
│   │   │   ├── writer.py       ← write SEG-Y
│   │   │   ├── headers.py      ← trace/binary/text headers
│   │   │   ├── geometry.py     ← inline/xline/cdp/offset mapping
│   │   │   └── utils.py        ← helpers (scaling, endian, etc.)
│   │   │
│   │   ├── attributes/
│   │   │   ├── __init__.py     ← PUBLIC API
│   │   │   │
│   │   │   ├── base/
│   │   │   │   ├── trace.py
│   │   │   │   └── window.py
│   │   │   │
│   │   │   ├── coherence2d.py
│   │   │   ├── coherence3d.py
│   │   │   ├── semblance2d.py
│   │   │   ├── semblance3d.py
│   │   │   │
│   │   │   ├── instantaneous_amplitude.py
│   │   │   ├── instantaneous_frequency.py
│   │   │   ├── instantaneous_phase.py
│   │   │   ├── cosine_instantaneous_phase.py
│   │   │   │
│   │   │   ├── amplitude_weighted_phase.py
│   │   │   ├── amplitude_weighted_frequency.py
│   │   │   ├── amplitude_weighted_cosine_phase.py
│   │   │   │
│   │   │   ├── integrate_traces.py
│   │   │   ├── integrate_absolute_amplitude.py
│   │   │   │
│   │   │   ├── derivative.py
│   │   │   ├── second_derivative.py
│   │   │   ├── derivative_instantaneous_amplitude.py
│   │   │   ├── second_derivative_inst_amp.py
│   │   │   │
│   │   │   └── quadrature.py
│   │   │
│   │   ├── plotting/
│   │   │   ├── __init__.py
│   │   │   ├── plot2d.py
│   │   │   └── plot3d.py
│   │   │
│   ├── well/
│   │   └── __init__.py
│   │
│   ├── map/
│   │   └── __init__.py
│   │
├── scripts/                    ← CLI / experiment scripts
├── pyproject.toml
└── README.md


