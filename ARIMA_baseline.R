library(ggplot2)
library(forecast)
library(tseries)
library(zoo)
library(xts)
library(gridExtra)

dfnew <- read.csv('hydro_norm.csv', header=TRUE, sep=',')

ggplot(dfnew, aes(x = Number, y = Solar)) + geom_line() +
 labs(x = 'Time', y='Solar_Gen(MWh)') +
 ggtitle('Variation of Solar with Time')

lat_ts_final = ts(dfnew[, c('Solar')], frequency = 24)
dec_electric = decompose(lat_ts_final)
dec_electric_graph <- autoplot(dec_electric)

nfitelectric <- auto.arima(lat_ts_final, stepwise=FALSE, approximation=FALSE)
nforelectric <- forecast(nfitelectric, h = 2181)
plot(nforelectric, xlab = "Number of Days", ylab = "SolarGen(MWh)")

write.csv(nforelectric, file = 'forecast_hydro_norm1.csv')

