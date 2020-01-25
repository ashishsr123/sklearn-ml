sleep_for_a_minute <- function() { Sys.sleep(60) }

start_time <- Sys.time()
sleep_for_a_minute()
end_time <- Sys.time()

end_time - start_time
# Time difference of 1.000327 mins

print(paste0("Current working dir: ", end_time - start_time))
