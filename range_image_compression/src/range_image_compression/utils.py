import rospy

def log_execution_time(out_path:str):
    def decorator(callback_func):
        def wrapper(*args, **kwargs):
            start_time = rospy.Time.now()
            result = callback_func(*args, **kwargs)
            end_time = rospy.Time.now()
            execution_time = end_time - start_time
            rospy.loginfo("Callback execution time: %s seconds", execution_time.to_sec())
        
            with open(out_path, 'a') as file:
                file.write(str(execution_time.to_sec()) + '\n')
        
            return result
        return wrapper
    return decorator
