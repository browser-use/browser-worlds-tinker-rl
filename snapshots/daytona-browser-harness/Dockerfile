FROM daytonaio/sandbox:0.8.0

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates chromium curl git python3-pip \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --break-system-packages --no-cache-dir \
      'browser-harness @ git+https://github.com/browser-use/browser-harness.git@f5eaf904b221dde0118eba1496961c3dc20fda88'

USER daytona
ENV HOME=/home/daytona
ENV BH_CHROME_PATH=/usr/bin/chromium
ENV CHROME_PATH=/usr/bin/chromium
ENV BH_NO_LIVE_OPEN=1
ENV BU_NO_LIVE_OPEN=1

USER root
RUN mkdir -p /opt/browser-rl-base \
    && printf '%s\n' \
      'browser_harness_commit=f5eaf904b221dde0118eba1496961c3dc20fda88' \
      'browser_harness_version=0.1.8' \
      > /opt/browser-rl-base/pin.txt \
    && chmod 0444 /opt/browser-rl-base/pin.txt

USER daytona
WORKDIR /home/daytona
