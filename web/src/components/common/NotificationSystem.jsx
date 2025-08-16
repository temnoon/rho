import React from 'react';

/**
 * NotificationSystem - Reusable notification components
 * 
 * Extracted from common notification patterns across components
 */

/**
 * Individual notification component
 */
export function Notification({ 
  id,
  message, 
  type = 'info', // info, success, warning, error
  duration = 5000,
  onClose = null,
  showCloseButton = true,
  style = {}
}) {
  const getNotificationStyle = () => {
    const baseStyle = {
      padding: '12px 16px',
      borderRadius: '6px',
      border: '1px solid',
      boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
      maxWidth: '350px',
      fontSize: '14px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      gap: '10px'
    };

    switch (type) {
      case 'error':
        return {
          ...baseStyle,
          background: '#ffebee',
          color: '#c62828',
          borderColor: '#f44336'
        };
      case 'warning':
        return {
          ...baseStyle,
          background: '#fff3e0',
          color: '#ef6c00',
          borderColor: '#ff9800'
        };
      case 'success':
        return {
          ...baseStyle,
          background: '#e8f5e9',
          color: '#2e7d32',
          borderColor: '#4caf50'
        };
      default: // info
        return {
          ...baseStyle,
          background: '#e3f2fd',
          color: '#1565c0',
          borderColor: '#2196f3'
        };
    }
  };

  const getIcon = () => {
    switch (type) {
      case 'error': return '❌';
      case 'warning': return '⚠️';
      case 'success': return '✅';
      default: return 'ℹ️';
    }
  };

  React.useEffect(() => {
    if (duration > 0 && onClose) {
      const timer = setTimeout(() => {
        onClose(id);
      }, duration);
      
      return () => clearTimeout(timer);
    }
  }, [duration, onClose, id]);

  return (
    <div style={{
      ...getNotificationStyle(),
      ...style
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <span>{getIcon()}</span>
        <span>{message}</span>
      </div>
      
      {showCloseButton && onClose && (
        <button
          onClick={() => onClose(id)}
          style={{
            background: 'none',
            border: 'none',
            fontSize: '16px',
            cursor: 'pointer',
            padding: '0',
            opacity: 0.7,
            ':hover': { opacity: 1 }
          }}
        >
          ×
        </button>
      )}
    </div>
  );
}

/**
 * Notification container that manages multiple notifications
 */
export function NotificationContainer({ 
  notifications = [],
  position = 'top-right', // top-right, top-left, bottom-right, bottom-left, top-center
  onRemove = null,
  maxNotifications = 5,
  style = {}
}) {
  const getPositionStyle = () => {
    const baseStyle = {
      position: 'fixed',
      zIndex: 1000,
      display: 'flex',
      flexDirection: 'column',
      gap: '10px',
      padding: '20px'
    };

    switch (position) {
      case 'top-left':
        return { ...baseStyle, top: 0, left: 0 };
      case 'top-center':
        return { 
          ...baseStyle, 
          top: 0, 
          left: '50%', 
          transform: 'translateX(-50%)',
          alignItems: 'center'
        };
      case 'bottom-left':
        return { ...baseStyle, bottom: 0, left: 0 };
      case 'bottom-right':
        return { ...baseStyle, bottom: 0, right: 0 };
      default: // top-right
        return { ...baseStyle, top: 0, right: 0 };
    }
  };

  // Limit number of notifications displayed
  const displayedNotifications = notifications.slice(-maxNotifications);

  if (displayedNotifications.length === 0) {
    return null;
  }

  return (
    <div style={{
      ...getPositionStyle(),
      ...style
    }}>
      {displayedNotifications.map(notification => (
        <Notification
          key={notification.id}
          {...notification}
          onClose={onRemove}
        />
      ))}
    </div>
  );
}

/**
 * Inline notification for in-content alerts
 */
export function InlineNotification({
  message,
  type = 'info',
  showIcon = true,
  dismissible = false,
  onDismiss = null,
  style = {}
}) {
  const getInlineStyle = () => {
    const baseStyle = {
      padding: '12px 16px',
      borderRadius: '6px',
      border: '1px solid',
      fontSize: '14px',
      display: 'flex',
      alignItems: 'center',
      gap: '10px'
    };

    switch (type) {
      case 'error':
        return {
          ...baseStyle,
          background: '#ffebee',
          color: '#c62828',
          borderColor: '#f44336'
        };
      case 'warning':
        return {
          ...baseStyle,
          background: '#fff3e0',
          color: '#ef6c00',
          borderColor: '#ff9800'
        };
      case 'success':
        return {
          ...baseStyle,
          background: '#e8f5e9',
          color: '#2e7d32',
          borderColor: '#4caf50'
        };
      default: // info
        return {
          ...baseStyle,
          background: '#e3f2fd',
          color: '#1565c0',
          borderColor: '#2196f3'
        };
    }
  };

  const getIcon = () => {
    if (!showIcon) return null;
    
    switch (type) {
      case 'error': return '❌';
      case 'warning': return '⚠️';
      case 'success': return '✅';
      default: return 'ℹ️';
    }
  };

  return (
    <div style={{
      ...getInlineStyle(),
      ...style
    }}>
      {showIcon && <span>{getIcon()}</span>}
      <span style={{ flex: 1 }}>{message}</span>
      {dismissible && onDismiss && (
        <button
          onClick={onDismiss}
          style={{
            background: 'none',
            border: 'none',
            fontSize: '16px',
            cursor: 'pointer',
            padding: '0',
            opacity: 0.7
          }}
        >
          ×
        </button>
      )}
    </div>
  );
}

/**
 * Agent message component (from NarrativeExplorer pattern)
 */
export function AgentMessage({ 
  children, 
  type = 'info', // info, question, success, warning
  icon = '🤖',
  agentName = 'Rho Agent',
  style = {}
}) {
  const getAgentStyle = () => {
    const baseStyle = {
      padding: '16px',
      margin: '16px 0',
      borderRadius: '8px',
      border: '2px solid',
      fontStyle: 'italic'
    };

    switch (type) {
      case 'question':
        return {
          ...baseStyle,
          backgroundColor: '#f0f8ff',
          borderColor: '#2196f3',
          color: '#1565c0'
        };
      case 'success':
        return {
          ...baseStyle,
          backgroundColor: '#f8fff0',
          borderColor: '#4caf50',
          color: '#2e7d32'
        };
      case 'warning':
        return {
          ...baseStyle,
          backgroundColor: '#fff8e1',
          borderColor: '#ff9800',
          color: '#ef6c00'
        };
      default:
        return {
          ...baseStyle,
          backgroundColor: '#f8f9fa',
          borderColor: '#dee2e6',
          color: '#495057'
        };
    }
  };

  return (
    <div style={{
      ...getAgentStyle(),
      ...style
    }}>
      <strong>{icon} {agentName}:</strong> {children}
    </div>
  );
}

/**
 * NotificationSystem - Combined component for easy import
 */
export const NotificationSystem = {
  Notification,
  NotificationContainer,
  InlineNotification,
  AgentMessage
};

export default NotificationSystem;