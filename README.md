# gta6-twitch-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/shangjf98-max/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/shangjf98-max/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts each remaining hour of a big launch's Twitch release day, as five quantiles, from the first hour's viewer count and the release-day curves of past launches supplied in the case file.

The model keeps only comparables that were already live at hour 1 (rolling-midnight launches, not simultaneous global unlocks), then fits one robust regression per hour: the log ratio of that hour's viewers to hour 1's, on the log of hour 1's viewers, with two Huber reweighting passes so one wild analogue cannot drag the shape. Small first hours leave room for the US unlock ramp; enormous first hours have less headroom, and the fitted slope grows from near zero in the small hours to about -0.45 by the evening. Quantiles are lognormal around the fitted median, with the regression's residual spread smoothed over neighbouring hours. Validated by leave-one-out over the 19 comparable-tier rolling launches in the kit: mean pinball loss 0.258 against 0.403 for the house persistence baseline. Deterministic, standard library only, no network.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=3h1pepd7&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
