module subs
! module that contains the core of the intergraion procedure of the OVM
! after changes run: 
!     
!    f2py -c  main.f90 -m main
!
! note you need to open a new python interpreter so that the module changes will be applicable

implicit none

contains

subroutine rk4(a,lambda,v0,h,L,N,m,wp,local_rho,local_flow,x,dotx,dt,ndotx,nx,ndeltax,nlocal_rho,nlocal_flow)
    !  integration scheme

    implicit none
    
    ! input variables 
    double precision,intent(in)   :: local_rho(:),   x(:),  dotx(:),   local_flow(:)
    double precision, intent(in)  :: a,lambda,v0,h,dt,L,wp
    integer, intent(in) :: N,m
    ! output variables 
    double precision, intent(out),dimension(size(local_rho)) :: nx, ndotx, ndeltax, nlocal_rho, nlocal_flow
    
    ! local variables
    double precision,dimension(size(local_rho)) :: k1,k2,k3,k4
    double precision,dimension(N+m) :: dummy_ndeltax,dummy_ndotx
    real(kind=8)  :: weightsum, boxsize, front_flow, weight
    integer :: i,j
    
    ! RK4 
    call acceleration_mcf(a,lambda,v0,h,local_rho,local_flow,dotx,k1)

    ndotx(:) = dotx(:) + k1*dt/2
    call acceleration_mcf(a,lambda,v0,h,local_rho,local_flow,ndotx,k2)

    ndotx(:) = dotx(:) + k2*dt/2
    call acceleration_mcf(a,lambda,v0,h,local_rho,local_flow,ndotx,k3)

    ndotx(:) = dotx(:) + k3*dt
    call acceleration_mcf(a,lambda,v0,h,local_rho,local_flow,ndotx,k4)

    ndotx(:) =  dotx(:) +  dt/6. * (k1 + 2*k2 + 2*k3 + k4)
    nx(:)      = x(:)  + ndotx * dt 
    
    ! calculate headways
    do i=1,size(local_rho)-1
       ndeltax(i) = nx(i+1)-nx(i)
    end do
    ndeltax(N) = nx(1)-nx(N)
    
    
    do i=1,N
       ! apply periodic boundary condition
       nx(i)      = mod(nx(i)+L,L)
       ndeltax(i) = mod(ndeltax(i)+L,L)
       ! prepare extended ndeltax array 
       dummy_ndeltax(i) = ndeltax(i)
       dummy_ndotx(i) = ndotx(i)
    end do
    
    ! extended arrays
    do i =1,m
       dummy_ndeltax(N+i) = ndeltax(i)
       dummy_ndotx(N+i) = ndotx(i)
    end do
    
    ! weighted local_rho and local_flow 
    do i=1,N
       boxsize = 0.0
       front_flow = 0.0
       weightsum = 0.0
       do j=1,m
          weight = exp(wp*(dble(m)-dble(j)))
          weightsum = weightsum + weight
          
          boxsize = boxsize + dummy_ndeltax(i+j-1) * weight 
          front_flow = front_flow + dummy_ndotx(i+j) * weight
       end do
       boxsize       = boxsize/weightsum !acctually here should be a "*m" but it would drop in the next step again
       nlocal_rho(i) = 1./boxsize
       nlocal_flow(i) = front_flow/weightsum
    end do

end subroutine



subroutine acceleration_mcf(a,lambda,v0,h,local_rho,local_flow,dotx,ddotx)
    ! Acceleration for the model OVM_rho_relax2J - that is similiar to the MCF introduced in Peng and Sun (2010)

    implicit none
    double precision,intent(in)   :: local_rho(:),dotx(:),local_flow(:)
    double precision, intent(in)  :: a,lambda,v0,h
    double precision, intent(out) :: ddotx(size(local_rho))
   
    double precision :: ovf(size(local_rho))

    call ovfunction(v0,h,local_rho(:),ovf(:))
 
    ddotx(:) =  a*(ovf(:) - dotx(:)) + lambda * (local_flow(:) - dotx(:)) 
  end subroutine acceleration_mcf


subroutine ovfunction(v0,h,local_rho,ovf)
     ! optimal velocity function as in Marschler et al (2014) but instead of using Delta_x here the inverse 
     ! local density is used. For just one head car this the inverse local_rho is the same as Delta_x

     implicit none
     double precision,intent(in)   :: local_rho(:)
     double precision, intent(in)  :: v0,h
     double precision, intent(out) :: ovf(size(local_rho))

     ovf(:)= v0*(tanh(1./local_rho(:) - h) + tanh(h))

   end subroutine 

subroutine annff(X,coef0,coef1,intercept0,intercept1,predict)
     implicit none
     double precision,intent(in)   :: X(:)
     double precision, intent(in)  :: coef0(:,:)
     double precision, intent(in)  :: coef1(:)
     double precision, intent(in)  :: intercept0(:)
     double precision, intent(in)  :: intercept1
     double precision :: hiddenout(size(coef1))
     double precision, intent(out) :: predict
     
     integer::i,j,imax,jmax
     imax = size(coef1)
     jmax = size(coef0(:,1))
    
     predict = 0.
     do i=1,imax
        hiddenout(i) = 0.
        do j=1,jmax
           hiddenout(i) = hiddenout(i) + X(j)*coef0(j,i)
        end do
        hiddenout(i) = logistic(hiddenout(i)+intercept0(i))
        predict = predict + hiddenout(i)*coef1(i) 
     end do
     predict = predict + intercept1

end subroutine annff



function logistic(x)

   implicit none 
   double precision :: logistic
   double precision, intent(in):: x
   
   logistic = 1./(1.+exp(-x))
   return

 end function

end module 
